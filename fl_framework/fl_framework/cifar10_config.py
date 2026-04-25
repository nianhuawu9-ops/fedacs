"""CIFAR-10 专用配置与快速验证脚本。

这个文件的用途：
1. 给 CIFAR-10 实验提供一组更稳的默认超参数。
2. 包含 [提速修改]：降低 local_epochs、降低 fraction、扩大 batch_size、降低 eval_every。
"""

from dataclasses import replace

import torch

from .config import FLConfig
from .models import get_model
from .client import FederatedClient
from .dataset import get_dataset, get_partition
from .servers.fedacs import FedACSServer


def get_cifar10_config() -> FLConfig:
    """返回一套适合 CIFAR-10 的默认配置。"""
    return FLConfig(
        dataset='cifar10',
        backbone='resnet8',
        num_clients=100,
        partition='dirichlet',
        alpha=0.3,
        rounds=100,
        
        # ========== [提速与防过拟合修改] ==========
        local_epochs=2,            # [修改] 从 5 降到 2，大幅减少本地计算量
        local_batch_size=128,      # [修改] 从 64 提升到 128，榨干 GPU 并行算力
        fraction=0.05,             # [修改] 从 0.1 降到 0.05，每轮只选 5 个客户端训练
        eval_every=5,              # [修改] 从 1 改为 5，每 5 轮才跑一次测试集推理
        # ==========================================

        local_lr=0.1,
        local_momentum=0.9,
        grad_clip=5.0,
        diagnostic_every=10,
        lr_scheduler='cosine',
        lr_warmup_rounds=0,
        min_local_lr=0.01,
        weight_decay=5e-4,
        heterogeneous_epochs=True,
        use_dp=False,
        
        # ========== [Server 端修复] ==========
        fedacs_server_opt='adam',  # 显式激活 Server 端的 Adam
        fedacs_server_lr=0.05,     # Server 端极度安全的学习率
        # =====================================
        
        fedacs_server_beta1=0.9,
        fedacs_server_beta2=0.999,
        fedacs_server_adam_eps=1e-4,
        fedacs_server_weight_decay=1e-4,
        fedacs_loss_ema_momentum=0.85,
        fedacs_fairness_window=5,
        fedacs_cooldown_rounds=0,
        fedacs_use_fairness=True,
        fedacs_use_update_norm=True,
        fedacs_use_drift=True,
        fedacs_use_cooldown=True,
        fedacs_use_quality_in_agg=True,
        
        target_accuracies=[0.70, 0.75, 0.80, 0.85, 0.90],
        data_dir='./data',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        seed=42,
        save_dir='./results/cifar10',
    )


def build_cifar10_experiment_cfg(base_cfg: FLConfig | None = None) -> FLConfig:
    """允许在通用配置基础上生成 CIFAR-10 专用配置。"""
    cfg = base_cfg or FLConfig()
    return replace(
        cfg,
        dataset='cifar10',
        backbone='resnet8',
        local_lr=0.1,
        local_epochs=2,            # [提速]
        local_batch_size=128,      # [提速]
        fraction=0.05,             # [提速]
        eval_every=5,              # [提速]
        grad_clip=5.0,
        weight_decay=5e-4,
        fedacs_server_opt='adam',  # [修复]
        fedacs_server_lr=0.05,     # [修复]
        fedacs_server_adam_eps=1e-4,
        fedacs_server_weight_decay=1e-4,
        fedacs_loss_ema_momentum=0.85,
        target_accuracies=[0.70, 0.75, 0.80, 0.85, 0.90],
    )


def verify_model(cfg: FLConfig | None = None):
    """快速检查模型是否能正常前向传播。"""
    cfg = cfg or get_cifar10_config()
    model = get_model(cfg).to(cfg.device)
    x = torch.randn(2, 3, 32, 32, device=cfg.device)
    y = model(x)
    print(f"[verify_model] output shape = {tuple(y.shape)}")
    print(f"[verify_model] finite = {bool(torch.isfinite(y).all())}")
    return y


if __name__ == '__main__':
    cfg = get_cifar10_config()
    print('[cifar10_config] 配置已生成:')
    print(f"Device: {cfg.device} (如果不是 cuda，请检查 PyTorch 环境！)")
    verify_model(cfg)