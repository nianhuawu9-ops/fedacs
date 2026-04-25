import argparse
import json
import os
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from .config import FLConfig
from .client import FederatedClient
from .dataset import get_dataset, get_partition
from .models import get_model
from .servers.fedacs import FedACSServer
from .servers.fedavg import FedAvgServer
from .servers.fednova import FedNovaServer
from .servers.fedprox import FedProxServer
from .servers.scaffold import SCAFFOLDServer

METHODS = {
    'FedAvg': (FedAvgServer, {}),
    'FedProx': (FedProxServer, {}),
    'SCAFFOLD': (SCAFFOLDServer, {}),
    'FedNova': (FedNovaServer, {}),
    'FedACS': (FedACSServer, {}),
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_config_from_args(args):
    """Build FLConfig from CLI arguments."""
    return FLConfig(
        dataset=args.dataset,
        backbone=args.backbone,
        num_clients=args.num_clients,
        partition=args.partition,
        alpha=args.alpha,
        num_shards_per_client=args.num_shards_per_client,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
        local_lr=args.local_lr,
        local_momentum=args.local_momentum,
        local_batch_size=args.local_batch_size,
        fraction=args.fraction,
        grad_clip=args.grad_clip,
        diagnostic_every=args.diagnostic_every,
        lr_scheduler=args.lr_scheduler,
        lr_warmup_rounds=args.lr_warmup_rounds,
        min_local_lr=args.min_local_lr,
        weight_decay=args.weight_decay,
        stragglers_frac=args.stragglers_frac,
        heterogeneous_epochs=not args.no_heterogeneous_epochs,
        use_dp=args.use_dp,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_max_grad_norm=args.dp_max_grad_norm,
        fedprox_mu=args.fedprox_mu,
        scaffold_lr=args.scaffold_lr,
        fedacs_alpha=args.fedacs_alpha,
        fedacs_beta=args.fedacs_beta,
        fedacs_epsilon=args.fedacs_epsilon,
        fedacs_loss_ema_momentum=args.fedacs_loss_ema_momentum,
        fedacs_selection_temperature=args.fedacs_selection_temperature,
        fedacs_fairness_window=args.fedacs_fairness_window,
        fedacs_cooldown_rounds=args.fedacs_cooldown_rounds,
        fedacs_use_fairness=not args.fedacs_no_fairness,
        fedacs_use_update_norm=not args.fedacs_no_update_norm,
        fedacs_use_drift=not args.fedacs_no_drift,
        fedacs_use_cooldown=not args.fedacs_no_cooldown,
        fedacs_use_quality_in_agg=not args.fedacs_no_quality_agg,
        eval_every=args.eval_every,
        target_accuracies=args.target_accuracies,
        data_dir=args.data_dir,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
    )


def build_clients(cfg, train_set, partition):
    return [
        FederatedClient(i, partition[i], train_set, cfg)
        for i in range(cfg.num_clients)
    ]


def run_single_experiment(cfg, method_name):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_set, test_set = get_dataset(cfg)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    partition = get_partition(cfg, train_set)

    if method_name not in METHODS:
        raise ValueError(f'Unknown method: {method_name}. Available: {list(METHODS.keys())}')

    ServerClass, extra_kwargs = METHODS[method_name]
    global_model = get_model(cfg)
    clients = build_clients(cfg, train_set, partition)
    server = ServerClass(global_model, clients, test_loader, cfg, **extra_kwargs)

    print(f"\n{'#'*60}")
    print(f"# 运行方法: {method_name}")
    print(f"{'#'*60}")

    summary = server.run()
    server.metrics.print_summary(method_name)

    save_path = os.path.join(cfg.save_dir, f'{method_name.lower()}_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in summary.items() if k != 'history'}, f, indent=2, default=str)
    print(f"\n结果已保存: {save_path}")

    if method_name == 'FedACS':
        summarize_fedacs_ablation({method_name: summary}, cfg.save_dir)

    return summary


def run_all_experiments(cfg):
    set_seed(cfg.seed)
    os.makedirs(cfg.save_dir, exist_ok=True)

    train_set, test_set = get_dataset(cfg)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
    partition = get_partition(cfg, train_set)

    all_results = {}

    for name, (ServerClass, extra_kwargs) in METHODS.items():
        print(f"\n{'#'*60}")
        print(f"# 运行方法: {name}")
        print(f"{'#'*60}")
        set_seed(cfg.seed)

        global_model = get_model(cfg)
        clients = build_clients(cfg, train_set, partition)
        server = ServerClass(global_model, clients, test_loader, cfg, **extra_kwargs)

        summary = server.run()
        server.metrics.print_summary(name)

        all_results[name] = summary

    save_path = os.path.join(cfg.save_dir, 'results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({k: {kk: vv for kk, vv in v.items() if kk != 'history'}
                   for k, v in all_results.items()}, f, indent=2, default=str)
    print(f"\n结果已保存: {save_path}")

    plot_all(all_results, cfg)
    summarize_fedacs_ablation(all_results, cfg.save_dir)
    return all_results


def run_batch_experiments(base_cfg, alphas, fractions):
    batch_dir = os.path.join(base_cfg.save_dir, 'batch_runs')
    os.makedirs(batch_dir, exist_ok=True)

    manifest = []
    summary_rows = []

    for alpha in alphas:
        for fraction in fractions:
            run_cfg = FLConfig(**base_cfg.__dict__)
            run_cfg.alpha = alpha
            run_cfg.fraction = fraction
            run_cfg.save_dir = os.path.join(batch_dir, f'alpha_{alpha}_frac_{fraction}')
            os.makedirs(run_cfg.save_dir, exist_ok=True)

            print(f"\n{'='*80}")
            print(f"批量实验开始: alpha={alpha}, fraction={fraction}")
            print(f"{'='*80}")
            results = run_all_experiments(run_cfg)

            compact = {
                'alpha': alpha,
                'fraction': fraction,
                'save_dir': run_cfg.save_dir,
                'methods': {
                    name: {
                        'final_top1': summary['final_top1'],
                        'final_loss': summary['final_loss'],
                        'final_fairness': summary['final_fairness'],
                        'final_utility': summary['final_utility'],
                        'total_time_s': summary['total_time_s'],
                        'total_comm_mb': summary['total_comm_mb'],
                        'rounds_to_target': summary['rounds_to_target'],
                    }
                    for name, summary in results.items()
                }
            }
            manifest.append(compact)
            summary_rows.append((alpha, fraction, results))

    manifest_path = os.path.join(batch_dir, 'batch_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"\n批量清单已保存: {manifest_path}")
    return manifest, summary_rows


def summarize_fedacs_ablation(results, save_dir):
    """Summarize FedACS ablation runs into a compact table and JSON."""
    if not results:
        return {}

    rows = []
    for name, summary in results.items():
        rows.append({
            'variant': name,
            'final_top1': summary.get('final_top1', 0.0),
            'best_top1': summary.get('best_top1', 0.0),
            'final_loss': summary.get('final_loss', 0.0),
            'final_fairness': summary.get('final_fairness', 0.0),
            'final_utility': summary.get('final_utility', 0.0),
            'total_comm_mb': summary.get('total_comm_mb', 0.0),
            'total_time_s': summary.get('total_time_s', 0.0),
            'final_selection_coverage_ratio': summary.get('final_selection_coverage_ratio', 0.0),
            'final_selection_entropy': summary.get('final_selection_entropy', 0.0),
        })

    rows = sorted(rows, key=lambda x: x['final_top1'], reverse=True)

    print(f"\n{'='*120}")
    print("FedACS 消融汇总")
    print(f"{'Variant':<28} {'Top1':>8} {'Best':>8} {'Loss':>8} {'Fairness':>10} {'Utility':>10} {'CommMB':>10} {'Time(s)':>10}")
    print(f"{'-'*120}")
    for r in rows:
        print(f"{r['variant']:<28} {r['final_top1']:>8.4f} {r['best_top1']:>8.4f} {r['final_loss']:>8.4f} {r['final_fairness']:>10.4f} {r['final_utility']:>10.4f} {r['total_comm_mb']:>10.1f} {r['total_time_s']:>10.1f}")
    print(f"{'='*120}")

    save_path = os.path.join(save_dir, 'fedacs_ablation_summary.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(rows, f, indent=2, default=str)
    print(f"FedACS 消融汇总已保存: {save_path}")
    return rows


def plot_all(results, cfg):
    COLORS = {
        'FedAvg':   ('#5B8DB8', '-'),
        'FedProx':  ('#E07B54', '--'),
        'SCAFFOLD': ('#6BAE75', '-.'),
        'FedNova':  ('#9B7EC8', ':'),
        'FedACS':   ('#D94F4F', '-'),
    }
    LINEWIDTH = {'FedACS': 2.5}

    save_comparison_convergence(results, cfg, COLORS, LINEWIDTH)
    save_comparison_fairness(results, cfg, COLORS, LINEWIDTH)
    save_comparison_targets(results, cfg, COLORS)

    print_comparison_table(results)


def save_comparison_convergence(results, cfg, colors, linewidth):
    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    fig.suptitle(
        f'FedACS vs Baselines — {cfg.dataset.upper()} '
        f'({cfg.partition}, α={cfg.alpha}, {cfg.num_clients} clients)',
        fontsize=14, fontweight='bold', y=1.03
    )

    ax1, ax2, ax3 = axes
    for name, summary in results.items():
        hist = summary['history']
        rounds = [r['round'] for r in hist]
        accs = [r['top1_acc'] for r in hist]
        losses = [r['loss'] for r in hist]
        comm = [r['cum_comm_mb'] for r in hist]
        c, ls = colors[name]
        lw = linewidth.get(name, 1.5)
        label = f'{name} (Ours)' if name == 'FedACS' else name
        ax1.plot(rounds, accs, color=c, linestyle=ls, linewidth=lw, label=label)
        ax2.plot(rounds, losses, color=c, linestyle=ls, linewidth=lw, label=name)
        ax3.plot(comm, accs, color=c, linestyle=ls, linewidth=lw, label=name)

    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Test Accuracy (Top-1)')
    ax1.set_title('(a) Convergence curve')
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Test Loss')
    ax2.set_title('(b) Loss convergence')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(alpha=0.3)

    ax3.set_xlabel('Cumulative Communication (MB)')
    ax3.set_ylabel('Test Accuracy (Top-1)')
    ax3.set_title('(c) Communication efficiency')
    ax3.legend(fontsize=8, loc='lower right')
    ax3.grid(alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(cfg.save_dir, 'comparison_convergence.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n收敛图已保存: {save_path}")
    plt.close(fig)


def save_comparison_fairness(results, cfg, colors, linewidth):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f'FedACS Fairness and Utility — {cfg.dataset.upper()} '
        f'({cfg.partition}, α={cfg.alpha}, {cfg.num_clients} clients)',
        fontsize=14, fontweight='bold', y=1.03
    )

    ax1, ax2 = axes
    for name, summary in results.items():
        c, ls = colors[name]
        lw = linewidth.get(name, 1.5)
        hist_f = summary['fairness_over_time']
        rounds_f = [r['round'] for r in hist_f]
        jain = [r['fairness_jain'] for r in hist_f]
        hist_u = summary['utility_over_time']
        rounds_u = [r['round'] for r in hist_u]
        util = [r['avg_client_utility'] for r in hist_u]
        ax1.plot(rounds_f, jain, color=c, linestyle=ls, linewidth=lw, label=name)
        ax2.plot(rounds_u, util, color=c, linestyle=ls, linewidth=lw, label=name)

    ax1.set_xlabel('Communication Rounds')
    ax1.set_ylabel('Weighted Jain Fairness')
    ax1.set_title('(d) Participation fairness over time')
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=8, loc='lower right')
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Communication Rounds')
    ax2.set_ylabel('Avg Client Utility')
    ax2.set_title('(e) Client utility over time')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    save_path = os.path.join(cfg.save_dir, 'comparison_fairness.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n公平性图已保存: {save_path}")
    plt.close(fig)


def save_comparison_targets(results, cfg, colors):
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(
        f'FedACS Target Achievement — {cfg.dataset.upper()} '
        f'({cfg.partition}, α={cfg.alpha}, {cfg.num_clients} clients)',
        fontsize=14, fontweight='bold', y=1.03
    )

    targets = cfg.target_accuracies
    x = np.arange(len(targets))
    width = 0.15
    method_names = list(results.keys())

    for i, name in enumerate(method_names):
        rtt = results[name]['rounds_to_target']
        vals = [rtt.get(t) or cfg.rounds for t in targets]
        c, _ = colors[name]
        lw = 2 if name == 'FedACS' else 1
        ax.bar(x + i * width, vals, width,
               label=name, color=c, linewidth=lw,
               edgecolor='black' if name == 'FedACS' else 'none')

    ax.set_xlabel('Target Accuracy')
    ax.set_ylabel('Rounds to achieve target')
    ax.set_title('(f) Rounds to target accuracy')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([f'{int(t*100)}%' for t in targets])
    ax.legend(fontsize=8, loc='upper left', ncol=1)
    ax.grid(alpha=0.3, axis='y')

    fig.tight_layout()
    save_path = os.path.join(cfg.save_dir, 'comparison_targets.png')
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n目标图已保存: {save_path}")
    plt.close(fig)


def print_comparison_table(results):
    print(f"\n{'='*96}")
    print(f"{'Method':<12} {'Top1':>8} {'Top5':>8} {'Loss':>8} {'CommMB':>10} {'Time(s)':>10} {'Fairness':>10} {'Utility':>10}")
    print(f"{'-'*96}")
    for name, summary in results.items():
        marker = ' (★)' if name == 'FedACS' else ''
        print(f"{name+marker:<16} "
              f"{summary['final_top1']:>8.4f} "
              f"{summary['final_top5']:>8.4f} "
              f"{summary['final_loss']:>8.4f} "
              f"{summary['total_comm_mb']:>10.1f} "
              f"{summary['total_time_s']:>10.1f} "
              f"{summary['final_fairness']:>10.4f} "
              f"{summary['final_utility']:>10.4f}")
    print(f"{'='*96}")


def parse_args():
    parser = argparse.ArgumentParser(description='Federated learning experiments')

    # Run mode / method entry
    parser.add_argument('--mode', choices=['single', 'all', 'batch'], default='single')
    parser.add_argument('--method', choices=list(METHODS.keys()), default='FedACS')
    parser.add_argument('--ablation', action='store_true', help='Run FedACS ablation variants in single mode')

    # Core experiment settings
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--backbone', default='auto', choices=['auto', 'legacy', 'resnet8'])
    parser.add_argument('--profile', default='auto', choices=['auto', 'mnist_balanced', 'cifar10_rescaled', 'cifar10_adamselect'])
    parser.add_argument('--num-clients', dest='num_clients', type=int, default=10)
    parser.add_argument('--partition', default='dirichlet', choices=['dirichlet', 'pathological', 'iid'])
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--num-shards-per-client', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--local-epochs', type=int, default=2)
    parser.add_argument('--local-lr', type=float, default=0.01)
    parser.add_argument('--local-momentum', type=float, default=0.9)
    parser.add_argument('--local-batch-size', type=int, default=32)
    parser.add_argument('--fraction', type=float, default=0.1)
    parser.add_argument('--diagnostic-every', dest='diagnostic_every', type=int, default=5)
    parser.add_argument('--grad-clip', type=float, default=10.0)
    parser.add_argument('--lr-scheduler', default='cosine', choices=['cosine', 'step', 'none'])
    parser.add_argument('--lr-warmup-rounds', type=int, default=0)
    parser.add_argument('--min-local-lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    parser.add_argument('--stragglers-frac', type=float, default=0.1)
    parser.add_argument('--no-heterogeneous-epochs', action='store_true')

    # DP
    parser.add_argument('--use-dp', action='store_true')
    parser.add_argument('--dp-noise-multiplier', type=float, default=0.5)
    parser.add_argument('--dp-max-grad-norm', type=float, default=1.0)

    # Method-specific hyperparams
    parser.add_argument('--fedprox-mu', type=float, default=0.01)
    parser.add_argument('--scaffold-lr', type=float, default=1.0)
    parser.add_argument('--fedacs-alpha', type=float, default=0.6)
    parser.add_argument('--fedacs-beta', type=float, default=0.3)
    parser.add_argument('--fedacs-epsilon', type=float, default=1e-8)
    parser.add_argument('--fedacs-loss-ema-momentum', type=float, default=0.8)
    parser.add_argument('--fedacs-selection-temperature', type=float, default=1.0)
    parser.add_argument('--fedacs-fairness-window', type=int, default=5)
    parser.add_argument('--fedacs-cooldown-rounds', type=int, default=0)
    parser.add_argument('--fedacs-no-fairness', action='store_true')
    parser.add_argument('--fedacs-no-update-norm', action='store_true')
    parser.add_argument('--fedacs-no-drift', action='store_true')
    parser.add_argument('--fedacs-no-cooldown', action='store_true')
    parser.add_argument('--fedacs-no-quality-agg', action='store_true')

    # Eval / storage
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--target-accuracies', type=float, nargs='+', default=[0.85, 0.9, 0.95, 0.97, 0.99])
    parser.add_argument('--data-dir', default='./data')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', default='./results')

    return parser.parse_args()


def run_fedacs_ablation(cfg):
    variants = {
        'full': dict(),
        'no_fairness': dict(fedacs_use_fairness=False),
        'no_update_norm': dict(fedacs_use_update_norm=False),
        'no_drift': dict(fedacs_use_drift=False),
        'no_cooldown': dict(fedacs_use_cooldown=False),
        'no_quality_agg': dict(fedacs_use_quality_in_agg=False),
    }

    ablation_results = {}
    for variant_name, overrides in variants.items():
        variant_cfg = FLConfig(**cfg.__dict__)
        for k, v in overrides.items():
            setattr(variant_cfg, k, v)
        variant_cfg.save_dir = os.path.join(cfg.save_dir, f'fedacs_ablation_{variant_name}')
        os.makedirs(variant_cfg.save_dir, exist_ok=True)
        print(f"\n{'='*80}")
        print(f"FedACS 消融开始: {variant_name}")
        print(f"{'='*80}")
        ablation_results[variant_name] = run_single_experiment(variant_cfg, 'FedACS')

    summarize_fedacs_ablation(ablation_results, cfg.save_dir)
    return ablation_results


if __name__ == '__main__':
    args = parse_args()
    cfg = build_config_from_args(args)
    if args.mode == 'batch':
        run_batch_experiments(cfg, alphas=[0.1, 0.3, 0.5, 1.0], fractions=[0.1, 0.3])
    elif args.mode == 'all':
        run_all_experiments(cfg)
    else:
        if args.method == 'FedACS' and args.ablation:
            run_fedacs_ablation(cfg)
        else:
            run_single_experiment(cfg, args.method)
