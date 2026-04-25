"""
client.py — 更新版，针对 CIFAR-10 优化本地训练

主要改动：
1. CIFAR-10 本地优化器改为 SGD + cosine annealing warmup（更稳定）
2. 梯度裁剪阈值收紧（CIFAR-10 用 5.0，MNIST 保持 10.0）
3. 本地训练增加 label smoothing（缓解 Non-IID 过拟合）
4. local_train 返回值加入 dataset 类型标记，供服务端自适应处理
"""
import copy
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset


class FederatedClient:
    def __init__(self, client_id, data_indices, dataset, cfg):
        self.id             = client_id
        self.cfg            = cfg
        self.device         = cfg.device
        self.dataset        = dataset
        self.data_indices   = data_indices
        self.data_size      = len(data_indices)
        self.is_cifar       = getattr(cfg, 'dataset', '').lower() == 'cifar10'

        # 追踪状态
        self.local_loss         = float('inf')
        self.local_loss_history = []
        self.grad_norm_history  = []
        self.last_selected_round = -1
        self.participation_count = 0

        # 系统异构
        self.compute_factor = (np.random.uniform(0.5, 1.5)
                               if cfg.heterogeneous_epochs else 1.0)

        # CIFAR-10 梯度裁剪更严格
        self._grad_clip = 5.0 if self.is_cifar else getattr(cfg, 'grad_clip', 10.0)

    # ── 数据加载 ────────────────────────────────────────────────

    def get_dataloader(self):
        subset = Subset(self.dataset, self.data_indices)
        return DataLoader(
            subset,
            batch_size=self.cfg.local_batch_size,
            shuffle=True,
            drop_last=False,
            num_workers=0,      # FL 场景下多进程反而慢
            pin_memory=(self.device == 'cuda'),
        )

    def get_local_epochs(self):
        base = self.cfg.local_epochs
        if self.cfg.heterogeneous_epochs:
            return max(1, int(base * self.compute_factor))
        return base

    def _add_dp_noise(self, model):
        if not self.cfg.use_dp:
            return
        for p in model.parameters():
            if p.grad is not None:
                noise = torch.randn_like(p.grad) * (
                    self.cfg.dp_noise_multiplier * self.cfg.dp_max_grad_norm
                )
                p.grad.add_(noise)

    def _compute_grad_norm(self, model):
        total_grad_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad_sq += p.grad.detach().norm().item() ** 2
        return math.sqrt(total_grad_sq)

    # ── 标准本地训练（FedAvg / FedACS 使用）────────────────────

    def local_train(self, global_model, **kwargs):
        """
        CIFAR-10 vs MNIST 的主要差异：
        - CIFAR-10：SGD lr=0.1 + cosine decay，label smoothing=0.1
        - MNIST：SGD lr=0.01 + momentum，无 label smoothing
        """
        model = copy.deepcopy(global_model).to(self.device)
        local_epochs = self.get_local_epochs()

        if self.is_cifar:
            # CIFAR-10：更大的初始 lr + cosine decay（参考 McMahan 2017 CIFAR 设置）
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=0.1,
                momentum=0.9,
                weight_decay=5e-4,
                nesterov=True,
            )
            # cosine annealing：让 lr 在本地 epochs 内从 0.1 衰减到 0.01
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=local_epochs, eta_min=0.01
            )
            # label smoothing 缓解 Non-IID 过拟合
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=self.cfg.local_lr,
                momentum=self.cfg.local_momentum,
                weight_decay=1e-4,
            )
            scheduler = None
            criterion = nn.CrossEntropyLoss()

        loader       = self.get_dataloader()
        model.train()
        total_loss   = 0.0
        num_batches  = 0
        grad_norms   = []

        for epoch in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)

                # 检测 nan/inf loss（CIFAR-10 初期可能出现）
                if not math.isfinite(loss.item()):
                    continue

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
                self._add_dp_noise(model)

                grad_norms.append(self._compute_grad_norm(model))

                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

            if scheduler is not None:
                scheduler.step()

        self.local_loss = total_loss / max(num_batches, 1)
        self.local_loss_history.append(self.local_loss)
        mean_grad_norm = float(np.mean(grad_norms)) if grad_norms else 0.0
        self.grad_norm_history.append(mean_grad_norm)
        self.participation_count += 1

        return model.state_dict(), {
            'loss':          self.local_loss,
            'grad_norm':     mean_grad_norm,
            'n_samples':     self.data_size,
            'actual_epochs': local_epochs,
            'client_id':     self.id,
            'dataset':       'cifar10' if self.is_cifar else 'mnist',
        }

    # ── FedProx 本地训练 ────────────────────────────────────────

    def local_train_fedprox(self, global_model, mu):
        model = copy.deepcopy(global_model).to(self.device)
        global_params = {n: p.detach().clone()
                         for n, p in global_model.named_parameters()}

        if self.is_cifar:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9,
                weight_decay=5e-4, nesterov=True)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.cfg.local_lr,
                momentum=self.cfg.local_momentum)
            criterion = nn.CrossEntropyLoss()

        loader       = self.get_dataloader()
        local_epochs = self.get_local_epochs()
        model.train()
        total_loss, num_batches = 0.0, 0

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                prox = sum(
                    ((p - global_params[n].to(self.device)) ** 2).sum()
                    for n, p in model.named_parameters()
                )
                loss = loss + (mu / 2) * prox
                if not math.isfinite(loss.item()):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        self.local_loss = total_loss / max(num_batches, 1)
        self.local_loss_history.append(self.local_loss)
        return model.state_dict(), {
            'loss': self.local_loss, 'n_samples': self.data_size, 'client_id': self.id
        }

    # ── SCAFFOLD 本地训练 ───────────────────────────────────────

    def local_train_scaffold(self, global_model, c_global, c_local):
        model = copy.deepcopy(global_model).to(self.device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=self.cfg.local_lr,
            momentum=0,   # SCAFFOLD 不使用 momentum
        )
        criterion    = nn.CrossEntropyLoss()
        loader       = self.get_dataloader()
        local_epochs = self.get_local_epochs()
        lr           = self.cfg.local_lr

        model.train()
        total_loss, num_batches = 0.0, 0

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if not math.isfinite(loss.item()):
                    continue
                loss.backward()
                # SCAFFOLD 控制变量修正
                for (name, param), ci, cg in zip(
                        model.named_parameters(), c_local, c_global):
                    if param.grad is not None:
                        param.grad.data.add_(-ci + cg)
                nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
                optimizer.step()
                total_loss  += loss.item()
                num_batches += 1

        steps = max(local_epochs * len(loader), 1)
        new_c_local = [
            ci.to(self.device) - cg.to(self.device)
            + (pw.to(self.device) - pp.to(self.device)) / (steps * lr)
            for ci, cg, (_, pw), (_, pp) in zip(
                c_local, c_global,
                global_model.named_parameters(),
                model.named_parameters())
        ]
        self.local_loss = total_loss / max(num_batches, 1)
        self.local_loss_history.append(self.local_loss)
        return model.state_dict(), new_c_local, {
            'loss': self.local_loss, 'n_samples': self.data_size, 'client_id': self.id
        }

    # ── FedNova 本地训练（修复双重 tau 缩放问题）──────────────────

    def local_train_fednova(self, global_model):
        model = copy.deepcopy(global_model).to(self.device)
        if self.is_cifar:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=0.1, momentum=0.9,
                weight_decay=5e-4, nesterov=True)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        else:
            optimizer = torch.optim.SGD(
                model.parameters(), lr=self.cfg.local_lr,
                momentum=self.cfg.local_momentum)
            criterion = nn.CrossEntropyLoss()

        loader       = self.get_dataloader()
        local_epochs = self.get_local_epochs()
        model.train()
        total_loss, total_steps = 0.0, 0

        for _ in range(local_epochs):
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                loss = criterion(model(x), y)
                if not math.isfinite(loss.item()):
                    continue
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), self._grad_clip)
                optimizer.step()
                total_loss  += loss.item()
                total_steps += 1

        tau_i = max(total_steps, 1)
        # nova_grad = (w_global - w_local) / tau_i
        # 服务端聚合时乘回 tau_i，还原为参数差
        nova_grad = {
            name: (gp.data.to(self.device) - lp.data.to(self.device)) / tau_i
            for (name, gp), (_, lp) in zip(
                global_model.named_parameters(), model.named_parameters())
        }

        self.local_loss = total_loss / max(total_steps, 1)
        self.local_loss_history.append(self.local_loss)
        return nova_grad, tau_i, {
            'loss': self.local_loss, 'n_samples': self.data_size, 'client_id': self.id
        }