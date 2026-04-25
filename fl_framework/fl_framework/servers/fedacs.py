"""
FedACS-v2: 终极修复版 + 极度 Non-IID 策略适配 + 防爆默认值
核心修复：
1. 移除了破坏向量方向的 torch.clamp。
2. 强制了默认优化器为 adam 和 lr=0.05，防止参数漏传导致爆炸。
3. 调整了公平性权重，增加了高质量客户端的话语权。
"""

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_server import BaseServer


# ─────────────────────────────────────────────
#  更强的 CIFAR-10 骨干网络（使用 GroupNorm 适配极度 Non-IID）
# ─────────────────────────────────────────────

class _ResBlock(nn.Module):
    """Pre-activation ResBlock，使用 GroupNorm 适应极度 Non-IID FL"""
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.gn1  = nn.GroupNorm(2, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.gn2  = nn.GroupNorm(2, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.gn1(x)))
        out = self.conv2(F.relu(self.gn2(out)))
        return out + self.shortcut(x)


class ResNet8(nn.Module):
    """
    ResNet-8：专为 CIFAR-10 FL 场景设计的轻量残差网络
    由于替换为 GroupNorm，完全免疫 dirichlet alpha=0.1 下的 BN 统计量崩溃
    """
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1, bias=False),
            nn.GroupNorm(2, 64),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _ResBlock(64, 64)
        self.layer2 = _ResBlock(64, 128, stride=2)
        self.layer3 = _ResBlock(128, 256, stride=2)
        self.pool   = nn.AdaptiveAvgPool2d(1)
        self.fc     = nn.Linear(256, num_classes)

        # 初始化，减少训练初期的不稳定
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)


# ─────────────────────────────────────────────
#  Per-client 状态
# ─────────────────────────────────────────────

@dataclass
class ClientAdaptiveState:
    selection_count:    int   = 0
    last_selected_round: int  = -1
    recent_selected:    List[int] = field(default_factory=list)
    loss_ema:           float = -1.0
    gain_ema:           float = 0.0
    stability_ema:      float = 0.0
    quality_ema:        float = 0.0
    last_loss_drop:     float = 0.0
    last_update_norm:   float = 0.0
    last_drift:         float = 0.0


# ─────────────────────────────────────────────
#  主服务端类
# ─────────────────────────────────────────────

class FedACSPlusServer(BaseServer):
    """
    FedACS-v2 (GroupNorm Fix + Drift Control)
    """

    def __init__(self, global_model, clients, test_loader, cfg):
        super().__init__(global_model, clients, test_loader, cfg)

        self.eps      = 1e-8
        self.is_cifar = getattr(cfg, 'dataset', '').lower() == 'cifar10'

        self.ema_mom  = 0.85 if self.is_cifar else 0.80
        self.fairness_window = int(getattr(cfg, 'fairness_window', 5))
        self.global_loss_ema: float = -1.0

        # ========== [终极防爆保护：设置极度安全的默认值] ==========
        default_server_lr = 0.05
        default_opt_mode  = 'adam'
        
        self.server_lr        = float(getattr(cfg, 'fedacs_server_lr', default_server_lr))
        self.server_opt_mode  = getattr(cfg, 'fedacs_server_opt', default_opt_mode)
        # =========================================================

        self.server_momentum  = float(getattr(cfg, 'fedacs_server_momentum', 0.0))
        self.server_adam_eps  = float(getattr(cfg, 'fedacs_server_adam_eps', 1e-4 if self.is_cifar else 1e-8))
        self.server_beta1     = float(getattr(cfg, 'fedacs_server_beta1', 0.9))
        self.server_beta2     = float(getattr(cfg, 'fedacs_server_beta2', 0.99 if self.is_cifar else 0.999))

        self.use_layerwise_clip   = True
        self.layerwise_clip_norm  = 5.0 if self.is_cifar else 5.0

        # [修改] 选择策略权重，适应极度 Non-IID
        if self.is_cifar:
            self.w_gain      = 1.0
            self.w_ucb       = 0.50
            self.w_fairness  = 0.30  # 降低强制公平性，防止漂移
            self.w_freshness = 0.20
            self.w_stability = 0.10
            self.w_cost      = 0.05
            self.exploit_r   = 0.70  # 增加贪心利用率
            self.fair_r      = 0.05  # 大幅降低无脑保底
            self.explore_r   = 0.25
            self.ucb_c       = 0.5
        else:
            self.w_gain      = 1.0
            self.w_ucb       = 0.4
            self.w_fairness  = 0.34
            self.w_freshness = 0.12
            self.w_stability = 0.08
            self.w_cost      = 0.08
            self.exploit_r   = 0.62
            self.fair_r      = 0.22
            self.explore_r   = 0.16
            self.ucb_c       = 0.8

        self.use_fairness = bool(getattr(cfg, 'fedacs_use_fairness', True))
        self.use_update_norm = bool(getattr(cfg, 'fedacs_use_update_norm', True))
        self.use_drift = bool(getattr(cfg, 'fedacs_use_drift', True))
        self.use_cooldown = bool(getattr(cfg, 'fedacs_use_cooldown', True))
        self.use_quality_in_agg = bool(getattr(cfg, 'fedacs_use_quality_in_agg', True))

        self.client_state: Dict[int, ClientAdaptiveState] = {
            c.id: ClientAdaptiveState() for c in clients
        }

        self.server_m: Dict[str, torch.Tensor] = {}
        self.server_v: Dict[str, torch.Tensor] = {}
        self.server_step = 0
        self.fairness_ema = 0.0

    @staticmethod
    def _sf(value, default: float = 0.0) -> float:
        try:
            v = float(value)
            return v if math.isfinite(v) else default
        except (TypeError, ValueError):
            return default

    def _ema(self, prev: float, new: float, mom: Optional[float] = None) -> float:
        m = mom if mom is not None else self.ema_mom
        prev = self._sf(prev, new)
        new  = self._sf(new, prev)
        return m * prev + (1.0 - m) * new

    def _state(self, cid: int) -> ClientAdaptiveState:
        if cid not in self.client_state:
            self.client_state[cid] = ClientAdaptiveState()
        return self.client_state[cid]

    def _is_valid_state_dict(self, state_dict: dict) -> bool:
        for tensor in state_dict.values():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                if not torch.isfinite(tensor).all():
                    return False
        return True

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        correct_top1 = correct_top5 = total = 0
        loss_sum = 0.0
        loss_batches = 0
        criterion = nn.CrossEntropyLoss()

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)

            if not torch.isfinite(logits).all():
                continue

            loss_val = criterion(logits, y)
            if torch.isfinite(loss_val):
                loss_sum += loss_val.item()
                loss_batches += 1

            pred = logits.argmax(dim=1)
            correct_top1 += (pred == y).sum().item()

            k = min(5, logits.size(1))
            _, top5_idx = logits.topk(k, dim=1)
            correct_top5 += (top5_idx == y.unsqueeze(1)).any(dim=1).sum().item()
            total += y.size(0)

        return {
            'top1_acc': correct_top1 / max(total, 1),
            'top5_acc': correct_top5 / max(total, 1),
            'loss': loss_sum / loss_batches if loss_batches > 0 else float('nan'),
        }

    def _ucb_bonus(self, cid: int) -> float:
        state = self._state(cid)
        t = max(self.current_round, 1)
        n = max(state.selection_count, 1)
        return self.ucb_c * math.sqrt(2.0 * math.log(t) / n)

    def _gain_signal(self, cid: int) -> float:
        state = self._state(cid)
        g_loss = self._sf(self.global_loss_ema, 2.3) if self.global_loss_ema >= 0 else 2.3
        local_loss = self._sf(state.loss_ema, g_loss) if state.loss_ema >= 0 else g_loss
        immediate = max(0.0, g_loss - local_loss)
        trend = max(0.0, self._sf(state.gain_ema, 0.0))
        return 0.65 * immediate + 0.35 * trend

    def _fairness_score(self, cid: int) -> float:
        state = self._state(cid)
        max_count = max((s.selection_count for s in self.client_state.values()), default=0)
        gap = max(0.0, max_count - state.selection_count)
        recent = state.recent_selected[-self.fairness_window:]
        recent_rate = sum(recent) / max(len(recent), 1)
        return gap + (1.0 - recent_rate)

    def _freshness_score(self, cid: int) -> float:
        state = self._state(cid)
        if state.last_selected_round < 0: return 1.0
        return max(0.0, self.current_round - state.last_selected_round) / max(self.current_round, 1)

    def _cost_signal(self, cid: int) -> float:
        state = self._state(cid)
        return 1.0 + 0.05 * state.selection_count + 0.03 * state.stability_ema

    def _selection_score(self, cid: int) -> float:
        state = self._state(cid)
        gain      = self._gain_signal(cid)
        ucb       = self._ucb_bonus(cid)
        fairness  = self._fairness_score(cid) if self.use_fairness else 0.0
        freshness = self._freshness_score(cid)
        cost      = self._cost_signal(cid)
        stab      = self._sf(state.stability_ema, 0.0) if self.use_update_norm else 0.0

        fair_term = math.sqrt(max(fairness, 0.0)) if self.is_cifar else fairness
        fresh_term = math.sqrt(max(freshness, 0.0)) if self.is_cifar else freshness
        gain_term = math.log1p(max(gain, 0.0)) if self.is_cifar else gain
        ucb_term = math.log1p(max(ucb, 0.0)) if self.is_cifar else ucb

        if self.is_cifar:
            raw = (self.w_gain * gain_term + self.w_fairness * fair_term + self.w_freshness * fresh_term + self.w_ucb * ucb_term - self.w_stability * stab) / (cost + self.eps)
        else:
            raw = (self.w_gain * gain + self.w_ucb * ucb + self.w_fairness * fairness + self.w_freshness * freshness - self.w_stability * stab) / (cost + self.eps)

        return float(math.tanh(raw)) if self.is_cifar else float(raw)

    def select_clients(self):
        k = max(1, int(len(self.clients) * self.cfg.fraction))
        if k >= len(self.clients): return list(self.clients)

        scores = {c.id: self._selection_score(c.id) for c in self.clients}
        ranked = sorted(self.clients, key=lambda c: scores[c.id], reverse=True)
        return self._layered_select(ranked, k)

    def _layered_select(self, ranked_clients, k: int):
        selected, selected_ids = [], set()

        def try_add(c):
            if c.id not in selected_ids and len(selected) < k:
                selected.append(c)
                selected_ids.add(c.id)
                return True
            return False

        exploit_k = max(1, int(k * self.exploit_r))
        for c in ranked_clients:
            if len(selected) >= exploit_k: break
            try_add(c)

        fair_r = getattr(self, 'fair_r', 0.05)
        fair_k = max(0, int(k * fair_r))

        if self.use_fairness and fair_k > 0:
            max_count = max((self._state(c.id).selection_count for c in self.clients), default=0)
            fair_pool = sorted(
                [c for c in self.clients if c.id not in selected_ids and self._state(c.id).selection_count <= max(0, max_count - 2)],
                key=lambda c: (self._state(c.id).selection_count, -self._gain_signal(c.id), -self._freshness_score(c.id)),
            )
            for c in fair_pool:
                if len(selected) >= exploit_k + fair_k: break
                try_add(c)

        remain = [c for c in self.clients if c.id not in selected_ids]
        if remain and len(selected) < k:
            extra_n = min(max(1, k - len(selected)), len(remain))
            for c in np.random.choice(remain, extra_n, replace=False).tolist():
                try_add(c)

        for c in ranked_clients:
            if len(selected) >= k: break
            try_add(c)

        return selected[:k]

    def _compute_delta(self, client_state_dict: dict) -> dict:
        global_sd = self.model.state_dict()
        return {
            k: client_state_dict[k].detach().clone().float().to(self.device) - global_sd[k].detach().clone().float().to(self.device)
            for k in client_state_dict if k in global_sd and torch.is_tensor(client_state_dict[k])
        }

    def _clip_delta_layerwise(self, delta: dict) -> dict:
        if not self.use_layerwise_clip: return delta
        clipped = {}
        for k, v in delta.items():
            if not torch.is_floating_point(v):
                clipped[k] = v
                continue
            norm = v.norm().item()
            if norm > self.layerwise_clip_norm and norm > self.eps:
                clipped[k] = v * (self.layerwise_clip_norm / norm)
            else:
                clipped[k] = v
        return clipped

    def _compute_quality(self, state_dict: dict, info: dict):
        cid = info['client_id']
        state = self._state(cid)
        g_loss = self._sf(self.global_loss_ema, 2.3) if self.global_loss_ema >= 0 else 2.3
        local_loss = self._sf(info.get('loss', g_loss), g_loss)
        loss_drop = max(0.0, g_loss - local_loss)

        update_norm = 0.0
        global_sd = self.model.state_dict()
        for key, tensor in state_dict.items():
            if key in global_sd and torch.is_floating_point(tensor):
                diff = tensor.float().to(self.device) - global_sd[key].float().to(self.device)
                update_norm += diff.pow(2).sum().item()
        update_norm = math.sqrt(update_norm)

        state.loss_ema = self._sf(info.get('loss', 2.3), 2.3) if state.loss_ema < 0 else self._ema(state.loss_ema, self._sf(info.get('loss', state.loss_ema), state.loss_ema))
        state.gain_ema = self._ema(state.gain_ema, loss_drop)
        state.stability_ema = self._ema(state.stability_ema, update_norm)

        stability_pen = 1.0 + (0.5 if self.is_cifar else 0.35) * state.stability_ema if self.use_update_norm else 1.0

        if self.is_cifar:
            quality = (0.7 * loss_drop + 0.3 * max(0.0, state.gain_ema)) / (stability_pen + self.eps)
        else:
            quality = (loss_drop + 0.5 * max(0.0, state.gain_ema)) / (stability_pen + self.eps)

        state.quality_ema = self._ema(state.quality_ema, max(0.0, quality))
        return max(0.0, quality)

    def aggregate_with_contrib(self, client_updates):
        if not client_updates: return
        valid_updates = [(sd, info) for sd, info in client_updates if self._is_valid_state_dict(sd)]
        if not valid_updates: return

        qualities = {info['client_id']: self._compute_quality(sd, info) for sd, info in valid_updates}
        total_n = max(sum(info['n_samples'] for _, info in valid_updates), self.eps)

        raw_w = {}
        for _, info in valid_updates:
            cid = info['client_id']
            n_w = info['n_samples'] / total_n
            # [修改] 拉大优质客户端的提权系数
            q_boost = 1.0 + (2.0 if self.is_cifar else 0.60) * qualities[cid] 
            raw_w[cid] = n_w * q_boost

        w_sum = max(sum(raw_w.values()), self.eps)
        first_sd = valid_updates[0][0]
        delta_avg = {k: torch.zeros_like(v.float().to(self.device)) for k, v in first_sd.items() if k in self.model.state_dict() and torch.is_tensor(v)}

        for sd, info in valid_updates:
            cid = info['client_id']
            w   = raw_w[cid] / w_sum
            delta = self._clip_delta_layerwise(self._compute_delta(sd))
            for key in delta_avg:
                if key in delta:
                    delta_avg[key] = delta_avg[key] + w * delta[key]

        if delta_avg:
            self._server_adam_update(delta_avg)

    def _server_adam_update(self, delta_avg: dict):
        """
        [终极修复]：无 Clamp 截断，纯粹的方向保留
        """
        self.server_step += 1
        if not self.server_m:
            self.server_m = {k: torch.zeros_like(v, device=self.device) for k, v in delta_avg.items()}
            self.server_v = {k: torch.full_like(v, self.server_adam_eps ** 2, device=self.device) for k, v in delta_avg.items()}

        updated_sd = {}
        model_sd = self.model.state_dict()

        for key, param in model_sd.items():
            if key not in delta_avg:
                updated_sd[key] = param.detach().clone().to(self.device)
                continue

            grad = -delta_avg[key].to(self.device)
            if not torch.isfinite(grad).all():
                updated_sd[key] = param.detach().clone().to(self.device)
                continue

            if self.server_opt_mode == 'fedavg':
                step = self.server_lr * delta_avg[key].to(self.device)
                updated_sd[key] = param.detach().clone().to(self.device) + step
                continue

            if self.server_opt_mode == 'fedavgm':
                if key not in self.server_m:
                    self.server_m[key] = torch.zeros_like(grad)
                self.server_m[key] = self.server_m[key] * self.server_momentum + (1.0 - self.server_momentum) * delta_avg[key].to(self.device)
                step = self.server_lr * self.server_m[key]
                updated_sd[key] = param.detach().clone().to(self.device) + step
                continue

            if key not in self.server_m:
                self.server_m[key] = torch.zeros_like(grad)
                self.server_v[key] = torch.full_like(grad, self.server_adam_eps ** 2)

            self.server_m[key] = self.server_beta1 * self.server_m[key] + (1.0 - self.server_beta1) * grad
            self.server_v[key] = self.server_beta2 * self.server_v[key] + (1.0 - self.server_beta2) * grad.pow(2)

            m_hat = self.server_m[key] / (1.0 - self.server_beta1 ** self.server_step)
            v_hat = self.server_v[key] / (1.0 - self.server_beta2 ** self.server_step)
            step = self.server_lr * m_hat / (torch.sqrt(v_hat) + self.server_adam_eps)

            if torch.isfinite(step).all():
                updated_sd[key] = param.detach().clone().to(self.device) - step
            else:
                updated_sd[key] = param.detach().clone().to(self.device)

        self.model.load_state_dict(updated_sd)

    def _update_participation(self, selected_ids: set):
        for c in self.clients:
            state = self._state(c.id)
            flag  = 1 if c.id in selected_ids else 0
            state.recent_selected.append(flag)
            if len(state.recent_selected) > max(10, self.fairness_window * 3):
                state.recent_selected.pop(0)

    def run(self):
        model_size_mb = self.get_model_size_mb()
        profile = 'cifar10' if self.is_cifar else 'mnist'
        print(f"\n[FedACS-v2/{profile}] server_lr={self.server_lr}, "
              f"opt={self.server_opt_mode}, momentum={self.server_momentum}")
        print(f"                  模型大小: {model_size_mb:.2f} MB")
        
        # 加上 evaluate 标识，避免被没打印的评估过程搞迷惑
        eval_flag = f"(每 {self.cfg.eval_every} 轮测试一次)"
        print(f"{'Round':>6} {'Top1':>8} {'Top5':>8} {'Loss':>8} {'CommMB':>8} {eval_flag}")

        for r in range(1, self.cfg.rounds + 1):
            self.current_round = r
            t0 = time.time()

            selected     = self.select_clients()
            selected_ids = {c.id for c in selected}
            self._update_participation(selected_ids)

            updates = []
            for c in selected:
                state = self._state(c.id)
                state.last_selected_round = r
                state.selection_count    += 1
                sd, info = c.local_train(self.model)
                if isinstance(sd, dict):
                    for k, v in list(sd.items()):
                        if torch.is_tensor(v) and torch.is_floating_point(v):
                            sd[k] = torch.nan_to_num(v, nan=0.0, posinf=1e4, neginf=-1e4)
                info['client_id'] = c.id
                updates.append((sd, info))

            self.aggregate_with_contrib(updates)
            
            # [提速核心]: 只在指定的轮数才全量评估
            if r % self.cfg.eval_every == 0 or r == self.cfg.rounds:
                eval_result = self.evaluate()
            else:
                eval_result = {
                    'top1_acc': float('nan'), 
                    'top5_acc': float('nan'), 
                    'loss': float('nan')
                }

            new_loss = self._sf(eval_result.get('loss', 2.3), 2.3)
            if not math.isnan(new_loss):
                if self.global_loss_ema < 0:
                    self.global_loss_ema = new_loss
                else:
                    self.global_loss_ema = self._ema(self.global_loss_ema, new_loss)

            counts     = [self._state(c.id).selection_count for c in self.clients]
            total_sel  = sum(counts)
            if total_sel > 0:
                jain = total_sel ** 2 / (len(counts) * sum(x * x for x in counts) + self.eps)
            else:
                jain = 0.0
            self.fairness_ema = self._ema(self.fairness_ema, jain, mom=0.8)

            comm_mb    = len(selected) * 2 * model_size_mb
            round_time = time.time() - t0
            self.metrics.record(r, eval_result, comm_mb, selected, round_time_s=round_time)

            if r % self.cfg.eval_every == 0 or r == self.cfg.rounds or r == 1:
                print(f"[Round {r:03d}] 完成 | "
                      f"Top1={eval_result['top1_acc']:.4f}, "
                      f"Top5={eval_result['top5_acc']:.4f}, "
                      f"Loss={eval_result['loss']:.4f}, "
                      f"CommMB={comm_mb:.2f}, "
                      f"RoundTime={round_time:.2f}s, "
                      f"Fairness={jain:.4f}")

        return self.metrics.get_summary()


# 保持向后兼容
FedACSServer = FedACSPlusServer