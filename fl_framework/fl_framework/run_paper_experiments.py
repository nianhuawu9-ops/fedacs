import argparse
import copy
import csv
import json
import os
from dataclasses import asdict

import torch

from .config import FLConfig
from .dataset import get_dataset, get_partition
from .main import METHODS, build_clients, run_all_experiments, set_seed, summarize_fedacs_ablation
from .models import get_model


BASE_DIR = os.path.join('.', 'results', 'paper_suite')
ORIGIN_DIR = os.path.join(BASE_DIR, 'origin_data')
DEFAULT_DATA_DIR = os.path.join('.', 'fl_framework', 'data')


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_cfg(shared_cfg: dict, overrides: dict) -> FLConfig:
    merged = copy.deepcopy(shared_cfg)
    merged.update(overrides)
    return FLConfig(**merged)


def compact_summary(summary: dict):
    return {
        'final_top1': summary.get('final_top1'),
        'final_top5': summary.get('final_top5'),
        'best_top1': summary.get('best_top1'),
        'final_loss': summary.get('final_loss'),
        'final_fairness': summary.get('final_fairness'),
        'final_participation_fairness': summary.get('final_participation_fairness'),
        'final_contribution_fairness': summary.get('final_contribution_fairness'),
        'final_selection_entropy': summary.get('final_selection_entropy'),
        'final_selection_coverage_ratio': summary.get('final_selection_coverage_ratio'),
        'final_utility': summary.get('final_utility'),
        'final_drift': summary.get('final_drift'),
        'auc_top1': summary.get('auc_top1'),
        'auc_fairness': summary.get('auc_fairness'),
        'peak_drop_top1': summary.get('peak_drop_top1'),
        'tail_std_top1': summary.get('tail_std_top1'),
        'tail_std_loss': summary.get('tail_std_loss'),
        'avg_drift': summary.get('avg_drift'),
        'total_time_s': summary.get('total_time_s'),
        'total_comm_mb': summary.get('total_comm_mb'),
        'rounds_to_target': summary.get('rounds_to_target'),
        'time_to_target': summary.get('time_to_target'),
        'comm_to_target': summary.get('comm_to_target'),
    }


def _origin_ready(v):
    if isinstance(v, (int, float, str)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _origin_ready(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_origin_ready(x) for x in v]
    return str(v)


def export_origin_result(path: str, row: dict):
    ensure_dir(os.path.dirname(path))
    exists = os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({k: _origin_ready(v) for k, v in row.items()})


EXPERIMENT_SUITES = {
    'cifar10_comparison': {
        'description': 'Main comparison on CIFAR-10 (cifar-10-batches-py) under non-IID partitioning.',
        'shared': {
            'dataset': 'cifar10',
            'backbone': 'resnet8',
            'num_clients': 20,
            'partition': 'dirichlet',
            'rounds': 50,
            'local_epochs': 3,
            'local_batch_size': 64,
            'seed': 42,
            'save_dir': os.path.join(BASE_DIR, 'cifar10_comparison'),
            'target_accuracies': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        },
        'grid': [
            {'alpha': 0.1, 'fraction': 0.3},
            {'alpha': 0.3, 'fraction': 0.3},
            {'alpha': 0.5, 'fraction': 0.3},
        ],
    },
    'fraction_sensitivity': {
        'description': 'Sensitivity analysis over participation fraction on CIFAR-10 (cifar-10-batches-py).',
        'shared': {
            'dataset': 'cifar10',
            'backbone': 'resnet8',
            'num_clients': 20,
            'partition': 'dirichlet',
            'alpha': 0.3,
            'rounds': 50,
            'local_epochs': 3,
            'local_batch_size': 64,
            'seed': 42,
            'save_dir': os.path.join(BASE_DIR, 'fraction_sensitivity'),
            'target_accuracies': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.96, 0.97, 0.98],
        },
        'grid': [
            {'fraction': 0.1},
            {'fraction': 0.2},
            {'fraction': 0.3},
            {'fraction': 0.5},
        ],
    },
    'communication_budget': {
        'description': 'Communication budget sensitivity on CIFAR-10 (cifar-10-batches-py).',
        'shared': {
            'dataset': 'cifar10',
            'backbone': 'resnet8',
            'num_clients': 20,
            'partition': 'dirichlet',
            'alpha': 0.3,
            'rounds': 50,
            'local_epochs': 3,
            'local_batch_size': 64,
            'fraction': 0.3,
            'seed': 42,
            'save_dir': os.path.join(BASE_DIR, 'communication_budget'),
            'target_accuracies': [0.60, 0.65, 0.70, 0.75, 0.80, 0.85],
        },
        'grid': [
            {'budget_tag': 'budget_small', 'fedacs_budget_scale': 0.75},
            {'budget_tag': 'budget_medium', 'fedacs_budget_scale': 1.00},
            {'budget_tag': 'budget_large', 'fedacs_budget_scale': 1.25},
        ],
    },
    'hardcase_deep': {
        'description': 'Stress test for strong heterogeneity and low participation on CIFAR-10 (cifar-10-batches-py).',
        'shared': {
            'dataset': 'cifar10',
            'backbone': 'resnet8',
            'num_clients': 20,
            'partition': 'dirichlet',
            'alpha': 0.1,
            'rounds': 100,
            'local_epochs': 1,
            'local_batch_size': 64,
            'seed': 42,
            'save_dir': os.path.join(BASE_DIR, 'hardcase_deep'),
            'target_accuracies': [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        },
        'grid': [
            {'fraction': 0.2},
            {'fraction': 0.3},
        ],
    },
    'mnist_ablation': {
        'description': 'Ablation study on MNIST.',
        'shared': {
            'dataset': 'mnist',
            'backbone': 'auto',
            'num_clients': 20,
            'partition': 'dirichlet',
            'alpha': 0.3,
            'rounds': 50,
            'local_epochs': 3,
            'local_batch_size': 64,
            'fraction': 0.3,
            'seed': 42,
            'save_dir': os.path.join(BASE_DIR, 'mnist_ablation'),
            'target_accuracies': [0.90, 0.92, 0.94, 0.96, 0.97, 0.98],
        },
        'grid': [
            {'method_tag': 'fedavg', 'method': 'FedAvg'},
            {'method_tag': 'fedprox', 'method': 'FedProx'},
            {'method_tag': 'fedacs_full', 'method': 'FedACS'},
            {'method_tag': 'fedacs_no_fairness', 'method': 'FedACS', 'fedacs_use_fairness': False, 'fedacs_use_benefit_fairness': False},
            {'method_tag': 'fedacs_no_budget', 'method': 'FedACS', 'fedacs_use_budget': False},
            {'method_tag': 'fedacs_no_update_norm', 'method': 'FedACS', 'fedacs_use_update_norm': False},
            {'method_tag': 'fedacs_no_drift', 'method': 'FedACS', 'fedacs_use_drift': False},
        ],
    },
}


def list_suites():
    print('Available experiment suites:')
    for name, spec in EXPERIMENT_SUITES.items():
        print(f'  - {name}: {spec["description"]}')


def _build_origin_row(suite_key: str, run_name: str, cfg: FLConfig, method_name: str, summary: dict):
    compact = compact_summary(summary)
    row = {
        'suite': suite_key,
        'run_name': run_name,
        'method': method_name,
        'dataset': cfg.dataset,
        'backbone': cfg.backbone,
        'partition': cfg.partition,
        'alpha': cfg.alpha,
        'fraction': cfg.fraction,
        'num_clients': cfg.num_clients,
        'rounds': cfg.rounds,
        'local_epochs': cfg.local_epochs,
        'seed': cfg.seed,
        'final_top1': compact.get('final_top1'),
        'final_top5': compact.get('final_top5'),
        'best_top1': compact.get('best_top1'),
        'final_loss': compact.get('final_loss'),
        'final_fairness': compact.get('final_fairness'),
        'final_participation_fairness': compact.get('final_participation_fairness'),
        'final_contribution_fairness': compact.get('final_contribution_fairness'),
        'final_selection_entropy': compact.get('final_selection_entropy'),
        'final_selection_coverage_ratio': compact.get('final_selection_coverage_ratio'),
        'final_utility': compact.get('final_utility'),
        'final_drift': compact.get('final_drift'),
        'auc_top1': compact.get('auc_top1'),
        'auc_fairness': compact.get('auc_fairness'),
        'peak_drop_top1': compact.get('peak_drop_top1'),
        'tail_std_top1': compact.get('tail_std_top1'),
        'tail_std_loss': compact.get('tail_std_loss'),
        'avg_drift': compact.get('avg_drift'),
        'total_time_s': compact.get('total_time_s'),
        'total_comm_mb': compact.get('total_comm_mb'),
        'rounds_to_target': compact.get('rounds_to_target'),
        'time_to_target': compact.get('time_to_target'),
        'comm_to_target': compact.get('comm_to_target'),
        'config_json': json.dumps(asdict(cfg), ensure_ascii=False, default=str),
    }
    return row


def run_named_suite(suite_key: str):
    if suite_key not in EXPERIMENT_SUITES:
        raise ValueError(f'Unknown suite: {suite_key}. Available: {list(EXPERIMENT_SUITES.keys())}')

    spec = EXPERIMENT_SUITES[suite_key]
    suite_dir = spec['shared']['save_dir']
    ensure_dir(suite_dir)
    ensure_dir(ORIGIN_DIR)
    manifest = []

    print(f"\n[Suite] {suite_key}: {spec['description']}")

    for item in spec['grid']:
        item_cfg = dict(item)
        method_name = item_cfg.pop('method', None)
        method_tag = item_cfg.pop('method_tag', None)
        run_name_parts = []
        if 'alpha' in item_cfg:
            run_name_parts.append(f"alpha_{item_cfg['alpha']}")
        if 'fraction' in item_cfg:
            run_name_parts.append(f"frac_{item_cfg['fraction']}")
        if 'budget_tag' in item_cfg:
            run_name_parts.append(str(item_cfg['budget_tag']))
        if method_tag:
            run_name_parts.append(str(method_tag))
        run_name = '_'.join(run_name_parts) if run_name_parts else 'run'

        cfg = build_cfg(spec['shared'], {**item_cfg, 'save_dir': os.path.join(suite_dir, run_name)})
        ensure_dir(cfg.save_dir)

        print(f"\n[Paper/{suite_key}] Running {run_name}")

        if method_name is None:
            results = run_all_experiments(cfg)
            manifest.append({
                'run_name': run_name,
                'config': asdict(cfg),
                'methods': {name: compact_summary(summary) for name, summary in results.items()},
            })
            for name, summary in results.items():
                origin_row = _build_origin_row(suite_key, run_name, cfg, name, summary)
                export_origin_result(os.path.join(ORIGIN_DIR, f'{suite_key}.csv'), origin_row)
        else:
            set_seed(cfg.seed)
            train_set, test_set = get_dataset(cfg)
            test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False)
            partition = get_partition(cfg, train_set)
            global_model = get_model(cfg)
            clients = build_clients(cfg, train_set, partition)

            if method_name not in METHODS:
                raise ValueError(f'Unknown method in suite: {method_name}')
            ServerClass, extra_kwargs = METHODS[method_name]
            server = ServerClass(global_model, clients, test_loader, cfg, **extra_kwargs)
            summary = server.run()
            server.metrics.print_summary(method_name)

            save_path = os.path.join(cfg.save_dir, f'{method_name.lower()}_results.json')
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump({k: v for k, v in summary.items() if k != 'history'}, f, indent=2, default=str)

            origin_row = _build_origin_row(suite_key, run_name, cfg, method_name, summary)
            export_origin_result(os.path.join(ORIGIN_DIR, f'{suite_key}.csv'), origin_row)

            manifest.append({
                'run_name': run_name,
                'method': method_name,
                'config': asdict(cfg),
                'summary': compact_summary(summary),
            })

    out_path = os.path.join(suite_dir, f'manifest_{suite_key}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f'Saved {suite_key} manifest to {out_path}')
    print(f'Origin data exported to: {os.path.join(ORIGIN_DIR, f"{suite_key}.csv")}')
    return manifest


def run_fedacs_ablation(cfg):
    """Run FedACS ablation on a supplied cfg."""
    variants = {
        'full': {},
        'no_fairness': {'fedacs_use_fairness': False, 'fedacs_use_benefit_fairness': False},
        'no_budget': {'fedacs_use_budget': False},
        'no_update_norm': {'fedacs_use_update_norm': False},
        'no_drift': {'fedacs_use_drift': False},
        'no_cooldown': {'fedacs_use_cooldown': False},
        'no_quality_agg': {'fedacs_use_quality_in_agg': False},
    }

    ablation_results = {}
    for variant_name, overrides in variants.items():
        variant_cfg = FLConfig(**cfg.__dict__)
        for k, v in overrides.items():
            setattr(variant_cfg, k, v)
        variant_cfg.save_dir = os.path.join(cfg.save_dir, f'fedacs_ablation_{variant_name}')
        ensure_dir(variant_cfg.save_dir)

        print(f"\n{'='*80}")
        print(f"FedACS ablation start: {variant_name}")
        print(f"{'='*80}")
        summary = run_single_experiment(variant_cfg, 'FedACS', suite_key='mnist_ablation', run_name=f'fedacs_{variant_name}')
        ablation_results[variant_name] = summary

    summarize_fedacs_ablation(ablation_results, cfg.save_dir)
    return ablation_results


def run_single_experiment(cfg, method_name, suite_key='manual', run_name='single_run'):
    set_seed(cfg.seed)
    ensure_dir(cfg.save_dir)
    ensure_dir(ORIGIN_DIR)

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
    print(f"# Running method: {method_name}")
    print(f"{'#'*60}")

    summary = server.run()
    server.metrics.print_summary(method_name)

    save_path = os.path.join(cfg.save_dir, f'{method_name.lower()}_results.json')
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump({k: v for k, v in summary.items() if k != 'history'}, f, indent=2, default=str)
    print(f'\nResults saved to: {save_path}')

    origin_row = _build_origin_row(suite_key, run_name, cfg, method_name, summary)
    export_origin_result(os.path.join(ORIGIN_DIR, f'{suite_key}.csv'), origin_row)

    if method_name == 'FedACS':
        summarize_fedacs_ablation({method_name: summary}, cfg.save_dir)

    return summary


def parse_args():
    parser = argparse.ArgumentParser(description='Federated learning paper experiments')
    parser.add_argument('--list-suites', action='store_true', help='List available experiment suites and exit')
    parser.add_argument('--suite', default='cifar10_comparison', choices=list(EXPERIMENT_SUITES.keys()))
    parser.add_argument('--mode', choices=['suite', 'single', 'all', 'ablation'], default='suite')
    parser.add_argument('--method', choices=list(METHODS.keys()), default='FedACS')
    parser.add_argument('--origin-dir', default=ORIGIN_DIR, help='Directory for Origin-ready CSV exports')

    parser.add_argument('--data-dir', default=DEFAULT_DATA_DIR, help='Dataset root directory')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save-dir', default='./results')

    # Optional overrides for single-run usage
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10'])
    parser.add_argument('--backbone', default='auto', choices=['auto', 'legacy', 'resnet8'])
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
    parser.add_argument('--use-dp', action='store_true')
    parser.add_argument('--dp-noise-multiplier', type=float, default=0.5)
    parser.add_argument('--dp-max-grad-norm', type=float, default=1.0)
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
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--target-accuracies', type=float, nargs='+', default=[0.85, 0.9, 0.95, 0.97, 0.99])

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.list_suites:
        list_suites()
        raise SystemExit(0)

    ORIGIN_DIR = args.origin_dir
    ensure_dir(BASE_DIR)
    ensure_dir(ORIGIN_DIR)

    for spec in EXPERIMENT_SUITES.values():
        spec['shared']['data_dir'] = args.data_dir

    if args.mode == 'suite':
        run_named_suite(args.suite)
    elif args.mode == 'all':
        for suite_key in ('cifar10_comparison', 'fraction_sensitivity', 'communication_budget', 'hardcase_deep', 'mnist_ablation'):
            run_named_suite(suite_key)
    elif args.mode == 'ablation':
        base = EXPERIMENT_SUITES['mnist_ablation']['shared']
        cfg = build_cfg(base, {'save_dir': os.path.join(BASE_DIR, 'mnist_ablation_manual')})
        cfg.data_dir = args.data_dir
        cfg.seed = args.seed
        cfg.device = args.device
        run_fedacs_ablation(cfg)
    else:
        cfg = FLConfig(
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
        if args.method == 'FedACS' and args.mode == 'ablation':
            run_fedacs_ablation(cfg)
        elif args.mode == 'all':
            run_all_experiments(cfg)
        else:
            run_single_experiment(cfg, args.method, suite_key='manual', run_name=f'{args.method.lower()}_manual')
