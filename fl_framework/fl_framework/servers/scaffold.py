import time
from collections import OrderedDict

import torch
from .base_server import BaseServer


class SCAFFOLDServer(BaseServer):
    """Karimireddy et al. 2020 — 控制变量消除客户端漂移"""

    def __init__(self, global_model, clients, test_loader, cfg):
        super().__init__(global_model, clients, test_loader, cfg)
        self.param_names = [name for name, _ in self.model.named_parameters()]
        self.c_global = OrderedDict((name, torch.zeros_like(param, device=self.device))
                                 for name, param in self.model.named_parameters())
        self.c_local = {
            c.id: OrderedDict((name, torch.zeros_like(param, device=self.device))
                             for name, param in self.model.named_parameters())
            for c in clients
        }

    def aggregate_scaffold(self, updates):
        if not updates:
            return

        total_n = max(sum(info['n_samples'] for _, _, info in updates), 1)

        current_sd = self.model.state_dict()
        agg_sd = OrderedDict()
        for key, tensor in current_sd.items():
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                agg_sd[key] = torch.zeros_like(tensor, device=self.device)
            else:
                agg_sd[key] = tensor.detach().clone() if torch.is_tensor(tensor) else tensor

        for state_dict, _, info in updates:
            w = info['n_samples'] / total_n
            for key, tensor in state_dict.items():
                if key in agg_sd and torch.is_tensor(tensor) and tensor.is_floating_point():
                    agg_sd[key].add_(w * tensor.to(self.device).float())

        self.model.load_state_dict(agg_sd, strict=False)

        lr_g = float(self.cfg.scaffold_lr)
        num_selected = max(len(updates), 1)
        for name in self.param_names:
            delta = torch.zeros_like(self.c_global[name], device=self.device)
            for _, new_c_local, _ in updates:
                delta.add_(new_c_local[name])
            self.c_global[name].add_((lr_g / num_selected) * delta)

    def run(self):
        model_size_mb = self.get_model_size_mb()
        print(f"\n[SCAFFOLD] 模型大小: {model_size_mb:.2f} MB")
        print(f"{'Round':>6} {'Top1':>8} {'Top5':>8} {'Loss':>8} {'CommMB':>8}")

        for r in range(1, self.cfg.rounds + 1):
            self.current_round = r
            round_start = time.time()
            selected = self.select_clients()
            updates = []
            c_global_list = list(self.c_global.values())
            for client in selected:
                c_local_list = list(self.c_local[client.id].values())
                state, new_c_local, info = client.local_train_scaffold(
                    self.model,
                    c_global_list,
                    c_local_list
                )
                self.c_local[client.id] = OrderedDict(zip(self.param_names, new_c_local))
                updates.append((state, self.c_local[client.id], info))

            self.aggregate_scaffold(updates)
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
                print(f"{r:>6} {eval_result['top1_acc']:>8.4f} "
                      f"{eval_result['top5_acc']:>8.4f} "
                      f"{eval_result['loss']:>8.4f} {comm_mb:>8.2f}")

        return self.metrics.get_summary()
