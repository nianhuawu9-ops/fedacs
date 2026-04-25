import torch
import numpy as np
from ..metrics import MetricsTracker


class BaseServer:
    """所有服务端实现的基类"""

    def __init__(self, global_model, clients, test_loader, cfg):
        self.model = global_model.to(cfg.device)
        self.clients = clients
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = cfg.device
        self.metrics = MetricsTracker(cfg)
        self.current_round = 0
        self.last_eval_result = None

    def select_clients(self):
        """子类重写此方法实现不同选择策略"""
        k = max(1, int(len(self.clients) * self.cfg.fraction))
        return np.random.choice(self.clients, k, replace=False).tolist()

    def aggregate(self, client_updates):
        """加权FedAvg聚合：w = Σ (n_k/n_total) · w_k"""
        total_n = sum(info['n_samples'] for _, info in client_updates)
        agg_dict = {}
        for key in client_updates[0][0]:
            agg_dict[key] = torch.zeros_like(
                client_updates[0][0][key], dtype=torch.float32
            )
            for state_dict, info in client_updates:
                w = info['n_samples'] / total_n
                agg_dict[key] += w * state_dict[key].float().to(self.device)
        self.model.load_state_dict(agg_dict)

    @torch.no_grad()
    def evaluate(self):
        """Evaluate every N rounds.

        - The first round always performs a real evaluation.
        - Afterwards, only every N rounds will run the full test set scan.
        - In-between rounds return the last real evaluation result.
        """
        eval_every = max(1, int(getattr(self.cfg, 'eval_every', 1)))
        need_eval = (self.current_round <= 1) or ((self.current_round % eval_every) == 0)

        if not need_eval:
            if self.last_eval_result is not None:
                return self.last_eval_result
            need_eval = True

        self.model.eval()
        correct_top1, correct_top5, total = 0, 0, 0
        loss_sum, loss_batches, nonfinite_batches = 0.0, 0, 0
        criterion = torch.nn.CrossEntropyLoss()

        for x, y in self.test_loader:
            x, y = x.to(self.device), y.to(self.device)
            logits = self.model(x)
            loss = criterion(logits, y)
            if torch.isfinite(loss):
                loss_sum += loss.item()
                loss_batches += 1
            else:
                nonfinite_batches += 1

            pred = logits.argmax(dim=1)
            correct_top1 += (pred == y).sum().item()

            _, top5 = logits.topk(min(5, logits.size(1)), dim=1)
            correct_top5 += (top5 == y.unsqueeze(1)).any(dim=1).sum().item()

            total += y.size(0)

        result = {
            'top1_acc': correct_top1 / max(total, 1),
            'top5_acc': correct_top5 / max(total, 1),
            'loss': loss_sum / max(loss_batches, 1),
        }
        if nonfinite_batches > 0:
            result['nonfinite_loss_batches'] = nonfinite_batches
        self.last_eval_result = result
        return result

    def get_model_size_mb(self):
        """计算模型参数量（MB）"""
        param_bytes = sum(p.numel() * p.element_size()
                          for p in self.model.parameters())
        return param_bytes / (1024 ** 2)

    def run(self):
        raise NotImplementedError