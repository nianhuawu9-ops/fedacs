import csv
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


BASE_RESULTS_DIR = os.path.join('.', 'results', 'paper_suite')
OUT_DIR = os.path.join(BASE_RESULTS_DIR, 'paper_assets')
ORIGIN_DIR = os.path.join(BASE_RESULTS_DIR, 'origin_exports')
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ORIGIN_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Times New Roman', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 180

METHOD_ORDER = ['FedAvg', 'FedProx', 'SCAFFOLD', 'FedNova', 'FedACS']
COLORS = {
    'FedAvg': '#4C78A8',
    'FedProx': '#F58518',
    'SCAFFOLD': '#54A24B',
    'FedNova': '#B279A2',
    'FedACS': '#E45756',
}


def load_manifest(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_csv(path, rows, fieldnames):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def flatten_main_grid(manifest):
    rows = []
    for item in manifest:
        for method, metrics in item['methods'].items():
            rows.append({
                'run_name': item['run_name'],
                'alpha': item['alpha'],
                'fraction': item['fraction'],
                'method': method,
                'final_top1': metrics['final_top1'],
                'final_loss': metrics['final_loss'],
                'final_fairness': metrics['final_fairness'],
                'final_participation_fairness': metrics.get('final_participation_fairness'),
                'final_contribution_fairness': metrics.get('final_contribution_fairness'),
                'final_selection_entropy': metrics.get('final_selection_entropy'),
                'final_selection_coverage_ratio': metrics.get('final_selection_coverage_ratio'),
                'final_utility': metrics['final_utility'],
                'final_drift': metrics.get('final_drift'),
                'auc_top1': metrics.get('auc_top1'),
                'auc_fairness': metrics.get('auc_fairness'),
                'peak_drop_top1': metrics.get('peak_drop_top1'),
                'tail_std_top1': metrics.get('tail_std_top1'),
                'tail_std_loss': metrics.get('tail_std_loss'),
                'avg_drift': metrics.get('avg_drift'),
                'total_time_s': metrics['total_time_s'],
                'total_comm_mb': metrics['total_comm_mb'],
                'rtt_095': metrics['rounds_to_target'].get('0.95'),
                'rtt_097': metrics['rounds_to_target'].get('0.97'),
                'rtt_098': metrics['rounds_to_target'].get('0.98'),
            })
    return rows


def flatten_ablation(manifest):
    rows = []
    for item in manifest:
        metrics = item['methods']['FedACS']
        rows.append({
            'run_name': item['run_name'],
            'variant': item['variant'],
            'alpha': item['alpha'],
            'fraction': item['fraction'],
            'final_top1': metrics['final_top1'],
            'final_loss': metrics['final_loss'],
            'final_fairness': metrics['final_fairness'],
            'final_participation_fairness': metrics.get('final_participation_fairness'),
            'final_contribution_fairness': metrics.get('final_contribution_fairness'),
            'final_selection_entropy': metrics.get('final_selection_entropy'),
            'final_selection_coverage_ratio': metrics.get('final_selection_coverage_ratio'),
            'final_utility': metrics['final_utility'],
            'final_drift': metrics.get('final_drift'),
            'auc_top1': metrics.get('auc_top1'),
            'auc_fairness': metrics.get('auc_fairness'),
            'peak_drop_top1': metrics.get('peak_drop_top1'),
            'tail_std_top1': metrics.get('tail_std_top1'),
            'tail_std_loss': metrics.get('tail_std_loss'),
            'avg_drift': metrics.get('avg_drift'),
            'total_time_s': metrics['total_time_s'],
            'total_comm_mb': metrics['total_comm_mb'],
            'rtt_095': metrics['rounds_to_target'].get('0.95'),
            'rtt_097': metrics['rounds_to_target'].get('0.97'),
            'rtt_098': metrics['rounds_to_target'].get('0.98'),
        })
    return rows


def flatten_hardcase(manifest):
    rows = []
    for item in manifest:
        for method, metrics in item['methods'].items():
            rows.append({
                'run_name': item['run_name'],
                'alpha': item.get('alpha'),
                'fraction': item.get('fraction'),
                'method': method,
                'final_top1': metrics['final_top1'],
                'final_loss': metrics['final_loss'],
                'final_participation_fairness': metrics.get('final_participation_fairness'),
                'final_contribution_fairness': metrics.get('final_contribution_fairness'),
                'final_selection_entropy': metrics.get('final_selection_entropy'),
                'final_selection_coverage_ratio': metrics.get('final_selection_coverage_ratio'),
                'peak_drop_top1': metrics.get('peak_drop_top1'),
                'tail_std_top1': metrics.get('tail_std_top1'),
                'tail_std_loss': metrics.get('tail_std_loss'),
                'total_time_s': metrics['total_time_s'],
                'total_comm_mb': metrics['total_comm_mb'],
            })
    return rows


def make_main_table(rows):
    out_path = os.path.join(OUT_DIR, 'table_main_results.csv')
    fieldnames = list(rows[0].keys())
    save_csv(out_path, rows, fieldnames)
    return out_path


def make_origin_export(path_hint, rows):
    if not rows:
        return None
    fieldnames = list(rows[0].keys())
    out_path = os.path.join(ORIGIN_DIR, path_hint)
    save_csv(out_path, rows, fieldnames)
    return out_path


def make_ablation_table(rows):
    out_path = os.path.join(OUT_DIR, 'table_ablation.csv')
    fieldnames = list(rows[0].keys())
    save_csv(out_path, rows, fieldnames)
    return out_path


def plot_accuracy_heatmaps(rows):
    grouped = defaultdict(dict)
    alphas = sorted({row['alpha'] for row in rows})
    fracs = sorted({row['fraction'] for row in rows})

    for row in rows:
        grouped[row['method']][(row['alpha'], row['fraction'])] = row['final_top1']

    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(4 * len(METHOD_ORDER), 4), constrained_layout=True)
    if len(METHOD_ORDER) == 1:
        axes = [axes]

    for ax, method in zip(axes, METHOD_ORDER):
        data = np.zeros((len(alphas), len(fracs)))
        for i, alpha in enumerate(alphas):
            for j, frac in enumerate(fracs):
                data[i, j] = grouped[method].get((alpha, frac), np.nan)
        im = ax.imshow(data, cmap='viridis', aspect='auto')
        ax.set_title(method)
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels(fracs)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels(alphas)
        ax.set_xlabel('Fraction')
        ax.set_ylabel('Alpha')
        for i in range(len(alphas)):
            for j in range(len(fracs)):
                ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label='Final Top-1')
    path = os.path.join(OUT_DIR, 'fig_main_accuracy_heatmaps.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_fairness_heatmaps(rows):
    grouped = defaultdict(dict)
    alphas = sorted({row['alpha'] for row in rows})
    fracs = sorted({row['fraction'] for row in rows})

    for row in rows:
        grouped[row['method']][(row['alpha'], row['fraction'])] = row['final_participation_fairness']

    fig, axes = plt.subplots(1, len(METHOD_ORDER), figsize=(4 * len(METHOD_ORDER), 4), constrained_layout=True)
    if len(METHOD_ORDER) == 1:
        axes = [axes]

    for ax, method in zip(axes, METHOD_ORDER):
        data = np.zeros((len(alphas), len(fracs)))
        for i, alpha in enumerate(alphas):
            for j, frac in enumerate(fracs):
                data[i, j] = grouped[method].get((alpha, frac), np.nan)
        im = ax.imshow(data, cmap='magma', aspect='auto', vmin=0, vmax=1)
        ax.set_title(method)
        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels(fracs)
        ax.set_yticks(range(len(alphas)))
        ax.set_yticklabels(alphas)
        ax.set_xlabel('Fraction')
        ax.set_ylabel('Alpha')
        for i in range(len(alphas)):
            for j in range(len(fracs)):
                ax.text(j, i, f'{data[i, j]:.3f}', ha='center', va='center', color='white', fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.8, label='Participation Jain Fairness')
    path = os.path.join(OUT_DIR, 'fig_main_fairness_heatmaps.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_hardcase_linechart(manifest):
    xs = [item['run_name'] for item in manifest]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

    for method in METHOD_ORDER:
        axes[0].plot(xs, [item['methods'][method]['final_top1'] for item in manifest], marker='o', color=COLORS[method], label=method)
        axes[1].plot(xs, [item['methods'][method]['final_participation_fairness'] for item in manifest], marker='o', color=COLORS[method], label=method)
        axes[2].plot(xs, [item['methods'][method].get('peak_drop_top1', 0.0) for item in manifest], marker='o', color=COLORS[method], label=method)

    axes[0].set_title('Hard-case Final Top-1')
    axes[0].set_ylabel('Final Top-1')
    axes[0].tick_params(axis='x', rotation=35)
    axes[0].grid(alpha=0.3)

    axes[1].set_title('Hard-case Participation Fairness')
    axes[1].set_ylabel('Participation Jain')
    axes[1].tick_params(axis='x', rotation=35)
    axes[1].grid(alpha=0.3)

    axes[2].set_title('Hard-case Peak Drop')
    axes[2].set_ylabel('Peak - Final Top-1')
    axes[2].tick_params(axis='x', rotation=35)
    axes[2].grid(alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=5, frameon=False)
    path = os.path.join(OUT_DIR, 'fig_hardcase_summary.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def plot_ablation_bars(rows):
    settings = sorted({(row['alpha'], row['fraction']) for row in rows})
    fig, axes = plt.subplots(len(settings), 3, figsize=(18, 4 * len(settings)), constrained_layout=True)
    if len(settings) == 1:
        axes = np.array([axes])

    for idx, setting in enumerate(settings):
        alpha, fraction = setting
        subset = [row for row in rows if row['alpha'] == alpha and row['fraction'] == fraction]
        subset = sorted(subset, key=lambda r: r['variant'])
        x = np.arange(len(subset))

        axes[idx, 0].bar(x, [row['final_top1'] for row in subset], color='#4C78A8')
        axes[idx, 0].set_title(f'Ablation Accuracy (alpha={alpha}, fraction={fraction})')
        axes[idx, 0].set_xticks(x)
        axes[idx, 0].set_xticklabels([row['variant'] for row in subset], rotation=30, ha='right')
        axes[idx, 0].set_ylabel('Final Top-1')
        axes[idx, 0].grid(alpha=0.3, axis='y')

        axes[idx, 1].bar(x, [row['final_participation_fairness'] for row in subset], color='#E45756')
        axes[idx, 1].set_title(f'Ablation Participation Fairness (alpha={alpha}, fraction={fraction})')
        axes[idx, 1].set_xticks(x)
        axes[idx, 1].set_xticklabels([row['variant'] for row in subset], rotation=30, ha='right')
        axes[idx, 1].set_ylabel('Participation Jain')
        axes[idx, 1].grid(alpha=0.3, axis='y')

        axes[idx, 2].bar(x, [row.get('peak_drop_top1', 0.0) for row in subset], color='#72B7B2')
        axes[idx, 2].set_title(f'Ablation Stability (alpha={alpha}, fraction={fraction})')
        axes[idx, 2].set_xticks(x)
        axes[idx, 2].set_xticklabels([row['variant'] for row in subset], rotation=30, ha='right')
        axes[idx, 2].set_ylabel('Peak Drop')
        axes[idx, 2].grid(alpha=0.3, axis='y')

    path = os.path.join(OUT_DIR, 'fig_ablation_bars.png')
    fig.savefig(path, bbox_inches='tight')
    plt.close(fig)
    return path


def write_latex_summary(main_rows, ablation_rows):
    fedacs_rows = [row for row in main_rows if row['method'] == 'FedACS']
    best_main = max(fedacs_rows, key=lambda row: row['final_top1']) if fedacs_rows else None
    best_fair = max(fedacs_rows, key=lambda row: row['final_participation_fairness']) if fedacs_rows else None
    best_ablation = max(ablation_rows, key=lambda row: row['final_top1']) if ablation_rows else None

    lines = [
        '%% Auto-generated summary for paper drafting',
        '\\section{Experimental Summary}',
    ]
    if best_main:
        lines.append(
            f"FedACS achieved its highest final Top-1 accuracy of {best_main['final_top1']:.4f} under alpha={best_main['alpha']} and fraction={best_main['fraction']}."
        )
    if best_fair:
        lines.append(
            f"The strongest participation fairness result for FedACS was {best_fair['final_participation_fairness']:.4f} under alpha={best_fair['alpha']} and fraction={best_fair['fraction']}."
        )
    if best_ablation:
        lines.append(
            f"In ablation experiments, variant {best_ablation['variant']} achieved the highest Top-1 accuracy of {best_ablation['final_top1']:.4f}."
        )

    path = os.path.join(OUT_DIR, 'paper_summary.tex')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return path


def main():
    main_manifest = load_manifest(os.path.join(BASE_RESULTS_DIR, 'main_grid', 'manifest_main_grid.json'))
    hard_manifest = load_manifest(os.path.join(BASE_RESULTS_DIR, 'hardcase_deep', 'manifest_hardcase.json'))
    ablation_manifest = load_manifest(os.path.join(BASE_RESULTS_DIR, 'ablation', 'manifest_ablation.json'))

    main_rows = flatten_main_grid(main_manifest)
    ablation_rows = flatten_ablation(ablation_manifest)

    hard_rows = flatten_hardcase(hard_manifest)

    assets = {
        'table_main': make_main_table(main_rows),
        'table_ablation': make_ablation_table(ablation_rows),
        'table_origin_main': make_origin_export('origin_main_results.csv', main_rows),
        'table_origin_ablation': make_origin_export('origin_ablation_results.csv', ablation_rows),
        'table_origin_hardcase': make_origin_export('origin_hardcase_results.csv', hard_rows),
        'fig_main_accuracy_heatmaps': plot_accuracy_heatmaps(main_rows),
        'fig_main_fairness_heatmaps': plot_fairness_heatmaps(main_rows),
        'fig_hardcase_summary': plot_hardcase_linechart(hard_manifest),
        'fig_ablation_bars': plot_ablation_bars(ablation_rows),
        'paper_summary': write_latex_summary(main_rows, ablation_rows),
    }

    with open(os.path.join(OUT_DIR, 'assets_manifest.json'), 'w', encoding='utf-8') as f:
        json.dump(assets, f, indent=2)

    print(json.dumps(assets, indent=2))


if __name__ == '__main__':
    main()
