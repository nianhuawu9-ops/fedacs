import time

import torch

from .base_server import BaseServer


class FedNovaServer(BaseServer):
    """Wang et al. 2020 — 归一化本地更新，缓解步数异质性偏差"""

    def aggregate_nova(self, nova_updates):
        """
        FedNova 风格的稳定聚合：
        - 先将每个客户端的更新按本地步数 tau 还原为“模型差”尺度；
        - 再按样本数加权求和；
        - 用全局学习率做一次同步更新。

        这样可以避免原实现里尺度被重复缩放，导致更新过小或过大。
        """
        if not nova_updates:
            return

        total_weight = sum(info['n_samples'] for _, _, info in nova_updates)
        total_weight = max(total_weight, 1e-8)

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                update = torch.zeros_like(param.data, device=self.device)
                for nova_grad, tau, info in nova_updates:
                    tau = max(int(tau), 1)
                    weight = info['n_samples'] / total_weight
                    update += weight * (nova_grad[name].to(self.device) * tau)
                param.data.add_(-self.cfg.local_lr * update)

    def run(self):
        model_size_mb = self.get_model_size_mb()
        print(f"\n[FedNova] 模型大小: {model_size_mb:.2f} MB")
        print(f"{'Round':>6} {'Top1':>8} {'Top5':>8} {'Loss':>8} {'CommMB':>8}")

        for r in range(1, self.cfg.rounds + 1):
            self.current_round = r
            round_start = time.time()
            selected = self.select_clients()
            updates = []
            for client in selected:
                nova_grad, tau, info = client.local_train_fednova(self.model)
                updates.append((nova_grad, tau, info))

            self.aggregate_nova(updates)
            eval_result = self.evaluate()

            comm_mb = len(selected) * 2 * model_size_mb
            round_time = time.time() - round_start
            self.metrics.record(r, eval_result, comm_mb, selected, round_time_s=round_time)

            print(
                f"[Round {r:03d}] 完成 | "
                f"Top1={eval_result['top1_acc']:.4f}, "
                f"Top5={eval_result['top5_acc']:.4f}, "
                f"Loss={eval_result['loss']:.4f}, "
                f"CommMB={comm_mb:.2f}, "
                f"RoundTime={round_time:.5f}s"
            )

            if r % 10 == 0 or r == 1:
                print(
                    f"{r:>6} {eval_result['top1_acc']:>8.4f} "
                    f"{eval_result['top5_acc']:>8.4f} "
                    f"{eval_result['loss']:>8.4f} {comm_mb:>8.2f}"
                )

        return self.metrics.get_summary()
