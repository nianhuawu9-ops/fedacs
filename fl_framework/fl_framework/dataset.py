import numpy as np
from torchvision import datasets, transforms

MNIST_MIRRORS = [
    'https://ossci-datasets.s3.amazonaws.com/mnist/',
    'https://storage.googleapis.com/cvdf-datasets/mnist/',
]


def _patch_mnist_download_sources():
    """Use more reliable MNIST mirrors when the default source is unavailable."""
    if hasattr(datasets.MNIST, 'mirrors'):
        datasets.MNIST.mirrors = MNIST_MIRRORS


def get_dataset(cfg):
    data_root = getattr(cfg, 'data_dir', './data')

    if cfg.dataset == 'mnist':
        _patch_mnist_download_sources()
        t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        try:
            train = datasets.MNIST(data_root, train=True, download=False, transform=t)
            test = datasets.MNIST(data_root, train=False, download=False, transform=t)
        except RuntimeError:
            print(f'MNIST not found in {data_root}, trying download=True once...')
            train = datasets.MNIST(data_root, train=True, download=True, transform=t)
            test = datasets.MNIST(data_root, train=False, download=True, transform=t)
    elif cfg.dataset == 'cifar10':
        t_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010))
        ])
        t_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                  (0.2023, 0.1994, 0.2010))
        ])
        try:
            train = datasets.CIFAR10(data_root, train=True, download=False, transform=t_train)
            test = datasets.CIFAR10(data_root, train=False, download=False, transform=t_test)
        except RuntimeError:
            print(f'CIFAR10 not found in {data_root}, trying download=True once...')
            train = datasets.CIFAR10(data_root, train=True, download=True, transform=t_train)
            test = datasets.CIFAR10(data_root, train=False, download=True, transform=t_test)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    return train, test


def partition_dirichlet(dataset, num_clients, alpha):
    """
    Dirichlet(alpha)分区 — 标准Non-IID基准
    alpha=0.1: 极端Non-IID; alpha=1.0: 近似IID
    """
    labels = np.array(dataset.targets)
    num_classes = len(np.unique(labels))
    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        cls_idx = np.where(labels == c)[0]
        np.random.shuffle(cls_idx)
        proportions = np.random.dirichlet([alpha] * num_clients)
        proportions = (proportions * len(cls_idx)).astype(int)
        proportions[-1] = len(cls_idx) - proportions[:-1].sum()  # 修正舍入误差
        splits = np.split(cls_idx, np.cumsum(proportions[:-1]))
        for i, idx in enumerate(splits):
            client_indices[i].extend(idx.tolist())

    return client_indices


def partition_pathological(dataset, num_clients, num_shards_per_client):
    """
    Pathological分区 — 每个客户端只有固定数量的类
    McMahan et al. (2017) 原始设置
    """
    labels = np.array(dataset.targets)
    total_shards = num_clients * num_shards_per_client
    shard_size = len(labels) // total_shards

    idx_sorted = np.argsort(labels)
    shards = [idx_sorted[i * shard_size:(i + 1) * shard_size]
              for i in range(total_shards)]
    np.random.shuffle(shards)

    client_indices = []
    for i in range(num_clients):
        assigned = shards[i * num_shards_per_client:(i + 1) * num_shards_per_client]
        client_indices.append(np.concatenate(assigned).tolist())

    return client_indices


def partition_iid(dataset, num_clients):
    """IID分区 — 均匀随机分配"""
    indices = np.random.permutation(len(dataset))
    chunks = np.array_split(indices, num_clients)
    return [chunk.tolist() for chunk in chunks]


def get_partition(cfg, dataset):
    if cfg.partition == 'dirichlet':
        return partition_dirichlet(dataset, cfg.num_clients, cfg.alpha)
    elif cfg.partition == 'pathological':
        return partition_pathological(dataset, cfg.num_clients, cfg.num_shards_per_client)
    elif cfg.partition == 'iid':
        return partition_iid(dataset, cfg.num_clients)
    raise ValueError(f"Unknown partition: {cfg.partition}")