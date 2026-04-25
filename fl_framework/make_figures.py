import os
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULT_PATH = os.path.join(BASE_DIR, 'fl_framework', 'results', 'results.json')
OUT_DIR = os.path.join(BASE_DIR, 'figures')
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def load_results():
    with open(RESULT_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)


def box(ax, x, y, w, h, text, fc, fs=11, weight='normal', ec='#2f2f2f'):
    patch = FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.05',
                           linewidth=1.1, edgecolor=ec, facecolor=fc)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontsize=fs, weight=weight)


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle='-|>', mutation_scale=12,
                                 linewidth=1.3, color='#444444'))


def save(fig, path):
    fig.savefig(path, dpi=240, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(path)


def framework_fig():
    path = os.path.join(OUT_DIR, 'framework.png')
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')

    title_c = '#1F4E79'
    c1, c2, c3, c4, c5 = '#D9E2F3', '#FCE4D6', '#E2F0D9', '#EDE7F6', '#DDEBF7'

    ax.text(7, 6.55, 'FedACS 总体框架', ha='center', va='center', fontsize=16, weight='bold', color=title_c)
    box(ax, 0.6, 5.2, 2.2, 0.9, '客户端状态\n损失 / 公平 / 效率', c1, fs=11, weight='bold')
    box(ax, 3.2, 5.2, 2.0, 0.9, '综合打分\n$ s_i^{(t)} $', c2, fs=12, weight='bold')
    box(ax, 5.6, 5.2, 2.6, 0.9, '分层选择\n高收益 / 公平补偿 / 低成本', c3, fs=11, weight='bold')
    box(ax, 8.6, 5.2, 2.2, 0.9, '本地训练\n客户端更新', c4, fs=11, weight='bold')
    box(ax, 11.2, 5.2, 2.2, 0.9, '加权聚合\n全局模型更新', c5, fs=11, weight='bold')

    box(ax, 2.5, 3.4, 2.6, 0.9, 'EMA 平滑\n稳定损失波动', '#FFF2CC', fs=11)
    box(ax, 6.0, 3.4, 2.8, 0.9, '预算约束\n选择客户端集合', '#F4B183', fs=11)
    box(ax, 10.0, 3.4, 2.5, 0.9, '贡献加权\n强化有效更新', '#C6E0B4', fs=11)

    box(ax, 2.0, 1.6, 10.0, 1.0, '输出：准确率、损失、通信量、达标轮次、公平性、客户端效用、漂移', '#EDE7F6', fs=12, weight='bold')

    arrow(ax, 2.8, 5.65, 3.2, 5.65)
    arrow(ax, 5.2, 5.65, 5.6, 5.65)
    arrow(ax, 8.2, 5.65, 8.6, 5.65)
    arrow(ax, 10.8, 5.65, 11.2, 5.65)

    arrow(ax, 4.5, 5.2, 4.5, 4.3)
    arrow(ax, 7.0, 5.2, 7.0, 4.3)
    arrow(ax, 9.7, 5.2, 9.7, 4.3)
    arrow(ax, 12.3, 5.2, 12.3, 4.3)

    arrow(ax, 5.8, 3.85, 6.0, 3.85)
    arrow(ax, 8.8, 3.85, 10.0, 3.85)

    arrow(ax, 11.25, 3.4, 11.25, 2.6)
    arrow(ax, 11.25, 2.6, 7.0, 2.6)
    arrow(ax, 7.0, 2.6, 7.0, 1.6)

    ax.text(7, 0.45, 'FedACS 在通信预算下实现高收益客户端优先、低参与客户端补偿与加权聚合的统一优化。',
            ha='center', va='center', fontsize=10, color='#666666')
    save(fig, path)


def flow_fig():
    path = os.path.join(OUT_DIR, 'method_flow.png')
    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    title_c = '#1F4E79'
    colors = ['#D9E2F3', '#FCE4D6', '#E2F0D9', '#EDE7F6', '#DDEBF7']
    ax.text(7, 4.95, 'FedACS 方法流程', ha='center', va='center', fontsize=16, weight='bold', color=title_c)

    box(ax, 0.6, 3.1, 2.2, 1.0, '1\n状态更新', colors[0], fs=12, weight='bold')
    box(ax, 3.1, 3.1, 2.2, 1.0, '2\n打分计算', colors[1], fs=12, weight='bold')
    box(ax, 5.6, 3.1, 2.4, 1.0, '3\n分层选择', colors[2], fs=12, weight='bold')
    box(ax, 8.3, 3.1, 2.2, 1.0, '4\n本地训练', colors[3], fs=12, weight='bold')
    box(ax, 10.8, 3.1, 2.4, 1.0, '5\n加权聚合', colors[4], fs=12, weight='bold')

    arrow(ax, 2.8, 3.6, 3.1, 3.6)
    arrow(ax, 5.3, 3.6, 5.6, 3.6)
    arrow(ax, 8.0, 3.6, 8.3, 3.6)
    arrow(ax, 10.5, 3.6, 10.8, 3.6)

    box(ax, 2.1, 1.2, 9.6, 0.95, '输出：新一轮全局模型 + 指标统计', '#EDE7F6', fs=12, weight='bold')
    arrow(ax, 12.0, 3.1, 12.0, 2.15)
    arrow(ax, 12.0, 2.15, 7.0, 2.15)
    arrow(ax, 7.0, 2.15, 7.0, 3.1)

    ax.text(7, 0.35, '从客户端状态到模型更新形成闭环优化。', ha='center', va='center', fontsize=10, color='#666666')
    save(fig, path)


def score_fig():
    path = os.path.join(OUT_DIR, 'scoring_layers.png')
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.2)
    ax.axis('off')

    title_c = '#1F4E79'
    ax.text(6, 4.65, 'FedACS 打分与分层选择', ha='center', va='center', fontsize=16, weight='bold', color=title_c)

    box(ax, 0.7, 3.5, 2.1, 0.9, '损失收益\n$ r_i^{(t)} $', '#D9E2F3', fs=12, weight='bold')
    box(ax, 3.3, 3.5, 2.1, 0.9, '公平补偿\n$ f_i^{(t)} $', '#FCE4D6', fs=12, weight='bold')
    box(ax, 5.9, 3.5, 2.1, 0.9, '效率项\n$ e_i^{(t)} $', '#E2F0D9', fs=12, weight='bold')
    box(ax, 8.5, 3.5, 2.7, 0.9, '综合得分\n$ s_i^{(t)} $', '#FFF2CC', fs=12, weight='bold')

    arrow(ax, 2.8, 3.95, 3.3, 3.95)
    arrow(ax, 5.4, 3.95, 5.9, 3.95)
    arrow(ax, 8.0, 3.95, 8.5, 3.95)

    box(ax, 1.0, 1.4, 2.2, 0.8, '高收益层', '#C6E0B4', fs=12, weight='bold')
    box(ax, 4.0, 1.4, 2.2, 0.8, '公平补偿层', '#FCE4D6', fs=12, weight='bold')
    box(ax, 7.0, 1.4, 2.2, 0.8, '低成本层', '#D9E2F3', fs=12, weight='bold')
    box(ax, 10.0, 1.25, 1.3, 1.1, '选中\n集合', '#B4C6E7', fs=12, weight='bold')

    arrow(ax, 10.0, 3.95, 10.65, 2.35)
    arrow(ax, 3.2, 1.8, 10.0, 1.8)
    arrow(ax, 6.2, 1.8, 10.0, 1.8)
    arrow(ax, 9.2, 1.8, 10.0, 1.8)

    ax.text(6, 0.35, '高收益、讲公平、低成本三者统一到同一选择机制中。', ha='center', va='center', fontsize=10, color='#666666')
    save(fig, path)


def save_single_fig(x, y, width, height, color, title, xlabel, ylabel, path, rotation=25):
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.bar(x, y, color=color, width=0.7)
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelrotation=rotation)
    fig.tight_layout()
    save(fig, path)


def _normalize_rounds_to_target(rtt):
    targets = [0.95, 0.96, 0.97, 0.98, 0.99]
    vals = []
    for t in targets:
        v = rtt.get(str(t))
        vals.append(v if v is not None else 10)
    return targets, vals


def results_fig(results):
    colors = {
        'FedAvg': '#5B8DB8',
        'FedProx': '#E07B54',
        'SCAFFOLD': '#6BAE75',
        'FedNova': '#9B7EC8',
        'FedACS': '#D94F4F',
    }

    for m, res in results.items():
        hist = res['history']
        fair_hist = res['fairness_over_time']
        util_hist = res['utility_over_time']

        save_single_fig(
            [r['round'] for r in hist],
            [r['top1_acc'] for r in hist],
            len(hist),
            1,
            colors.get(m, '#333333'),
            f'{m} - Top1 Accuracy',
            'Round',
            'Accuracy',
            os.path.join(OUT_DIR, f'{m.lower()}_top1.png'),
            rotation=0,
        )
        save_single_fig(
            [r['round'] for r in hist],
            [r['loss'] for r in hist],
            len(hist),
            1,
            colors.get(m, '#333333'),
            f'{m} - Test Loss',
            'Round',
            'Loss',
            os.path.join(OUT_DIR, f'{m.lower()}_loss.png'),
            rotation=0,
        )
        save_single_fig(
            [r['cum_comm_mb'] for r in hist],
            [r['top1_acc'] for r in hist],
            len(hist),
            1,
            colors.get(m, '#333333'),
            f'{m} - Communication Efficiency',
            'Cumulative Comm (MB)',
            'Accuracy',
            os.path.join(OUT_DIR, f'{m.lower()}_comm_efficiency.png'),
            rotation=0,
        )
        save_single_fig(
            [r['round'] for r in fair_hist],
            [r['fairness_jain'] for r in fair_hist],
            len(fair_hist),
            1,
            colors.get(m, '#333333'),
            f'{m} - Participation Fairness',
            'Round',
            'Jain Fairness',
            os.path.join(OUT_DIR, f'{m.lower()}_fairness.png'),
            rotation=0,
        )
        save_single_fig(
            [r['round'] for r in util_hist],
            [r['avg_client_utility'] for r in util_hist],
            len(util_hist),
            1,
            colors.get(m, '#333333'),
            f'{m} - Client Utility',
            'Round',
            'Utility',
            os.path.join(OUT_DIR, f'{m.lower()}_utility.png'),
            rotation=0,
        )

        targets, vals = _normalize_rounds_to_target(res['rounds_to_target'])
        save_single_fig(
            [f'{int(t*100)}%' for t in targets],
            vals,
            len(targets),
            1,
            colors.get(m, '#333333'),
            f'{m} - Rounds to Target Accuracy',
            'Target Accuracy',
            'Rounds',
            os.path.join(OUT_DIR, f'{m.lower()}_rounds_to_target.png'),
            rotation=0,
        )

    # 额外保存一份“单图总览”，但仍然是分开的文件，便于直接引用
    overview_path = os.path.join(OUT_DIR, 'comparison_overview.txt')
    with open(overview_path, 'w', encoding='utf-8') as f:
        f.write('Generated separate figures for each metric and method.\n')
        for m in results:
            f.write(f'- {m}: top1, loss, comm_efficiency, fairness, utility, rounds_to_target\n')
    print(overview_path)


if __name__ == '__main__':
    results = load_results()
    framework_fig()
    flow_fig()
    score_fig()
    results_fig(results)
