import numpy as np
from collections import defaultdict


class MetricsTracker:
    """
    论文友好的指标追踪器：
    1. top1_acc / loss / best_top1
    2. rounds / time / communication to target accuracy
    3. participation_jain_fairness      — 基于累计参与次数的标准 Jain 公平性
    4. contribution_jain_fairness       — 基于累计贡献代理的 Jain 公平性
    5. selection_coverage_ratio         — 被选中过的客户端覆盖率
    6. selection_entropy                — 选择分布归一化熵
    7. client_drift                     — 参与客户端局部损失离散度
    8. stability metrics                — AUC、peak drop、末窗口标准差
    9. utility_over_time                — 作为辅助诊断指标保留，不作为主论文指标
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.records = []
        self.cum_comm_mb = 0.0
        self.cum_time_s = 0.0
        self.rounds_to_target = {t: None for t in cfg.target_accuracies}
        self.time_to_target = {t: None for t in cfg.target_accuracies}
        self.comm_to_target = {t: None for t in cfg.target_accuracies}
        self.participation_counts = defaultdict(int)
        self.contribution_scores = defaultdict(float)
        self.client_loss_history = defaultdict(list)
        self.client_selection_history = defaultdict(list)
        self.client_update_norm_history = defaultdict(list)
        self.utility_history = []
        self.fairness_history = []

    @staticmethod
    def _safe_float(x, default=0.0):
        try:
            x = float(x)
        except (TypeError, ValueError):
            return default
        if not np.isfinite(x):
            return default
        return x

    def _compute_client_utilities(self, selected_clients, eval_result):
        """保留 utility 作为诊断型指标，不作为主论文公平性定义。"""
        eval_loss = self._safe_float(eval_result.get('loss', 0.0), 0.0)
        max_size = max(c.data_size for c in selected_clients) if selected_clients else 1

        utilities = []
        for c in selected_clients:
            local_loss = self._safe_float(c.local_loss, eval_loss)
            grad_norm = self._safe_float(np.mean(c.grad_norm_history[-1:]) if c.grad_norm_history else 0.0, 0.0)
            participation_gain = max(0.0, eval_loss - local_loss)
            data_efficiency = 1.0 - (c.data_size / max_size)
            utility = (
                0.50 * participation_gain +
                0.30 * grad_norm +
                0.20 * data_efficiency
            )
            utility = self._safe_float(utility, 0.0)
            utilities.append((c.id, utility, local_loss))
        return utilities

    def _compute_jain(self, values):
        values = np.asarray(values, dtype=np.float64)
        values = np.maximum(values, 0.0)
        n = values.size
        if n == 0:
            return 0.0
        total = values.sum()
        if total <= 1e-12:
            return 0.0
        denom = n * np.sum(values ** 2)
        if denom <= 1e-12:
            return 0.0
        return float((total ** 2) / denom)

    def _compute_selection_entropy(self):
        n = int(self.cfg.num_clients)
        if n <= 1:
            return 0.0
        counts = np.array([self.participation_counts.get(i, 0) for i in range(n)], dtype=np.float64)
        total = counts.sum()
        if total <= 1e-12:
            return 0.0
        probs = counts / total
        probs = probs[probs > 0]
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        return float(entropy / np.log(n))

    def _compute_selection_coverage_ratio(self):
        n = int(self.cfg.num_clients)
        if n <= 0:
            return 0.0
        covered = sum(1 for i in range(n) if self.participation_counts.get(i, 0) > 0)
        return float(covered / n)

    def _compute_participation_fairness(self):
        n = int(self.cfg.num_clients)
        counts = [self.participation_counts.get(i, 0) for i in range(n)]
        return self._compute_jain(counts)

    def _compute_contribution_fairness(self):
        n = int(self.cfg.num_clients)
        scores = [self.contribution_scores.get(i, 0.0) for i in range(n)]
        return self._compute_jain(scores)

    def record(self, round_num, eval_result, comm_mb, selected_clients, round_time_s=0.0):
        self.cum_comm_mb += comm_mb
        self.cum_time_s += round_time_s

        for c in selected_clients:
            self.participation_counts[c.id] += 1
            self.client_loss_history[c.id].append(self._safe_float(getattr(c, 'local_loss', 0.0), 0.0))
            self.client_selection_history[c.id].append(round_num)
            if hasattr(c, 'grad_norm_history') and c.grad_norm_history:
                self.client_update_norm_history[c.id].append(self._safe_float(c.grad_norm_history[-1], 0.0))

        utilities = self._compute_client_utilities(selected_clients, eval_result)
        for cid, utility, _ in utilities:
            self.contribution_scores[cid] += max(0.0, utility)

        round_utility = float(np.mean([u for _, u, _ in utilities])) if utilities else 0.0
        self.utility_history.append({
            'round': round_num,
            'avg_client_utility': round_utility,
            'selected_clients': [cid for cid, _, _ in utilities],
        })

        participation_fairness = self._compute_participation_fairness()
        contribution_fairness = self._compute_contribution_fairness()
        selection_entropy = self._compute_selection_entropy()
        selection_coverage_ratio = self._compute_selection_coverage_ratio()
        selection_gini = 1.0 - participation_fairness if participation_fairness <= 1.0 else 0.0

        self.fairness_history.append({
            'round': round_num,
            'fairness_jain': participation_fairness,
            'participation_fairness': participation_fairness,
            'contribution_fairness': contribution_fairness,
            'selection_entropy': selection_entropy,
            'selection_coverage_ratio': selection_coverage_ratio,
            'selection_gini': selection_gini,
            'avg_client_utility': round_utility,
        })

        local_losses = [self._safe_float(c.local_loss, 0.0) for c in selected_clients]
        drift = float(np.std(local_losses)) if local_losses else 0.0

        record = {
            'round': round_num,
            'top1_acc': self._safe_float(eval_result.get('top1_acc', 0.0), 0.0),
            'top5_acc': self._safe_float(eval_result.get('top5_acc', 0.0), 0.0),
            'loss': self._safe_float(eval_result.get('loss', 0.0), 0.0),
            'cum_comm_mb': self.cum_comm_mb,
            'cum_time_s': self.cum_time_s,
            'fairness_jain': participation_fairness,
            'participation_fairness': participation_fairness,
            'contribution_fairness': contribution_fairness,
            'selection_entropy': selection_entropy,
            'selection_coverage_ratio': selection_coverage_ratio,
            'selection_gini': selection_gini,
            'avg_client_utility': round_utility,
            'client_drift': drift,
        }
        self.records.append(record)

        for target in self.cfg.target_accuracies:
            if record['top1_acc'] >= target and self.rounds_to_target[target] is None:
                self.rounds_to_target[target] = round_num
                self.time_to_target[target] = self.cum_time_s
                self.comm_to_target[target] = self.cum_comm_mb

    def get_summary(self):
        if not self.records:
            return {}
        final = self.records[-1]
        accs = np.array([r['top1_acc'] for r in self.records], dtype=np.float64)
        losses = np.array([r['loss'] for r in self.records], dtype=np.float64)
        fair = np.array([r['participation_fairness'] for r in self.records], dtype=np.float64)
        drifts = np.array([r['client_drift'] for r in self.records], dtype=np.float64)
        best_acc = float(np.max(accs))
        stability_window = min(5, len(accs))

        selection_entropies = np.array([r['selection_entropy'] for r in self.records], dtype=np.float64)
        coverage_ratios = np.array([r['selection_coverage_ratio'] for r in self.records], dtype=np.float64)
        gini_values = np.array([r['selection_gini'] for r in self.records], dtype=np.float64)

        return {
            'final_top1': final['top1_acc'],
            'final_top5': final['top5_acc'],
            'best_top1': best_acc,
            'final_loss': final['loss'],
            'total_comm_mb': self.cum_comm_mb,
            'total_time_s': self.cum_time_s,
            'rounds_to_target': self.rounds_to_target,
            'time_to_target': self.time_to_target,
            'comm_to_target': self.comm_to_target,
            'final_fairness': final['participation_fairness'],
            'final_participation_fairness': final['participation_fairness'],
            'final_contribution_fairness': final['contribution_fairness'],
            'final_selection_entropy': final['selection_entropy'],
            'final_selection_coverage_ratio': final['selection_coverage_ratio'],
            'final_selection_gini': final['selection_gini'],
            'final_utility': final['avg_client_utility'],
            'final_drift': final['client_drift'],
            'auc_top1': float(np.mean(accs)),
            'auc_fairness': float(np.mean(fair)),
            'auc_selection_entropy': float(np.mean(selection_entropies)),
            'auc_selection_coverage_ratio': float(np.mean(coverage_ratios)),
            'auc_selection_gini': float(np.mean(gini_values)),
            'peak_drop_top1': float(best_acc - final['top1_acc']),
            'tail_std_top1': float(np.std(accs[-stability_window:])),
            'tail_std_loss': float(np.std(losses[-stability_window:])),
            'avg_drift': float(np.mean(drifts)),
            'history': self.records,
            'fairness_over_time': self.fairness_history,
            'utility_over_time': self.utility_history,
        }

    def print_summary(self, method_name):
        s = self.get_summary()
        print(f"\n{'='*60}")
        print(f"[{method_name}] 实验结果汇总")
        print(f"  最终Top-1准确率:          {s['final_top1']:.4f}")
        print(f"  最佳Top-1准确率:          {s['best_top1']:.4f}")
        print(f"  Top-1 曲线AUC(均值):      {s['auc_top1']:.4f}")
        print(f"  最终测试损失:             {s['final_loss']:.4f}")
        print(f"  总通信量:                 {s['total_comm_mb']:.1f} MB")
        print(f"  总耗时:                   {s['total_time_s']:.1f} s")
        print(f"  参与公平性(Jain):         {s['final_participation_fairness']:.4f}")
        print(f"  贡献公平性(Jain):         {s['final_contribution_fairness']:.4f}")
        print(f"  选择覆盖率:               {s['final_selection_coverage_ratio']:.4f}")
        print(f"  选择熵:                   {s['final_selection_entropy']:.4f}")
        print(f"  选择Gini:                 {s['final_selection_gini']:.4f}")
        print(f"  平均客户端漂移:           {s['avg_drift']:.4f}")
        print(f"  峰值回落(Peak Drop):      {s['peak_drop_top1']:.4f}")
        print(f"  末窗口Top-1标准差:        {s['tail_std_top1']:.4f}")
        print(f"  首次达到目标准确率:")
        for t, r in s['rounds_to_target'].items():
            tt = s['time_to_target'][t]
            ct = s['comm_to_target'][t]
            print(
                f"    {int(t*100)}%: {'未达到' if r is None else f'第{r}轮'}"
                f" | time={'--' if tt is None else f'{tt:.1f}s'}"
                f" | comm={'--' if ct is None else f'{ct:.1f}MB'}"
            )
        print(f"{'='*60}")
