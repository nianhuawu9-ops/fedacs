import time
from .base_server import BaseServer


class FedAvgServer(BaseServer):
    """McMahan et al. 2017 — 随机客户端选择"""

    def run(self):
        model_size_mb = self.get_model_size_mb()
        print(f"\n[FedAvg] 模型大小: {model_size_mb:.2f} MB")
        print(f"{'Round':>6} {'Top1':>8} {'Top5':>8} {'Loss':>8} {'CommMB':>8}")

        for r in range(1, self.cfg.rounds + 1):
            self.current_round = r
            round_start = time.time()

            selected = self.select_clients()
            updates = []
            for client in selected:
                state, info = client.local_train(self.model)
                updates.append((state, info))

            self.aggregate(updates)
            eval_result = self.evaluate()

            comm_mb = len(selected) * 2 * model_size_mb  # 下发+上传
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
                print(f"{r:>6} {eval_result['top1_acc']:>8.4f} "
                      f"{eval_result['top5_acc']:>8.4f} "
                      f"{eval_result['loss']:>8.4f} {comm_mb:>8.2f}")

        return self.metrics.get_summary()
