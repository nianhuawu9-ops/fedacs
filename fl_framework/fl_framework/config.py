from dataclasses import dataclass, field
from typing import List
import torch

@dataclass
class FLConfig:
    # ---- 数据 ----
    dataset: str = 'mnist'          # 'mnist' | 'cifar10'
    backbone: str = 'auto'          # 'auto' | 'resnet8' | 'legacy'
    num_clients: int = 10
    partition: str = 'dirichlet'    # 'dirichlet' | 'pathological' | 'iid'
    alpha: float = 0.3              # Dirichlet集中参数（越小越Non-IID）
    num_shards_per_client: int = 2  # Pathological分区每客户端类别数

    # ---- 训练 ----
    rounds: int = 100
    local_epochs: int = 5
    local_lr: float = 0.01
    local_momentum: float = 0.9
    local_batch_size: int = 32
    fraction: float = 0.1           # 每轮参与客户端比例
    grad_clip: float = 10.0         # 梯度裁剪阈值
    diagnostic_every: int = 5       # 诊断指标低频计算间隔（batch级）
    lr_scheduler: str = 'cosine'    # 'cosine' | 'step' | 'none'
    lr_warmup_rounds: int = 0       # 前几轮warmup（可选）
    min_local_lr: float = 1e-4      # cosine下限学习率
    weight_decay: float = 5e-4      # CIFAR-10建议使用

    # ---- 系统异构 ----
    stragglers_frac: float = 0.1    # 掉线/延迟客户端比例
    heterogeneous_epochs: bool = True  # 模拟不同客户端本地轮数不同

    # ---- 差分隐私 ----
    use_dp: bool = False
    dp_noise_multiplier: float = 0.5
    dp_max_grad_norm: float = 1.0

    # ---- 方法超参 ----
    fedprox_mu: float = 0.01        # FedProx近端项系数
    scaffold_lr: float = 1.0        # SCAFFOLD服务端LR
    fedacs_alpha: float = 0.6       # FedACS：损失差异权重
    fedacs_beta: float = 0.3        # FedACS：历史贡献衰减
    fedacs_epsilon: float = 1e-8    # FedACS：数值稳定项
    fedacs_loss_ema_momentum: float = 0.8   # FedACS：损失EMA动量
    fedacs_selection_temperature: float = 1.0  # FedACS：选择温度
    fedacs_fairness_window: int = 5         # FedACS：公平性滑动窗口
    fedacs_cooldown_rounds: int = 0         # FedACS：客户端冷却轮数
    fedacs_explore_min: float = 0.05        # FedACS++：最小探索比例
    fedacs_explore_max: float = 0.50        # FedACS++：最大探索比例
    fedacs_fairness_boost: float = 0.35      # FedACS++：公平补偿强度
    fedacs_coverage_weight: float = 1.0      # FedACS++：覆盖奖励权重
    fedacs_redundancy_weight: float = 0.25   # FedACS++：重复选择惩罚权重
    fedacs_drift_weight: float = 0.15        # FedACS++：漂移惩罚权重
    fedacs_cost_weight: float = 0.15         # FedACS++：效率项权重
    fedacs_budget_scale: float = 1.10        # FedACS++：预算放大系数
    fedacs_recent_window: int = 5           # FedACS++：最近选择窗口大小
    fedacs_use_fairness: bool = True        # FedACS：是否启用公平性项
    fedacs_use_update_norm: bool = True     # FedACS：是否启用 update_norm 项
    fedacs_use_drift: bool = True           # FedACS：是否启用漂移惩罚
    fedacs_use_cooldown: bool = True        # FedACS：是否启用冷却机制
    fedacs_use_quality_in_agg: bool = True   # FedACS：是否启用质量加权聚合

    # ---- 指标 ----
    metrics_window_size: int = 5            # 稳定性统计窗口
    metrics_report_per_round: bool = True   # 是否逐轮记录增强指标

    # ---- 评估 ----
    eval_every: int = 1
    target_accuracies: List[float] = field(
        default_factory=lambda: [0.85, 0.9, 0.95, 0.97, 0.99])  # rounds-to-X%指标阈值

    # ---- 其他 ----
    data_dir: str = './data'
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    save_dir: str = './results'
    