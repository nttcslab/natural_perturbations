import numpy as np
from scipy.stats import mannwhitneyu
import pandas as pd
import read_log
from omegaconf import DictConfig


def print_line(name: str, header: str, cfg_list: list[DictConfig], df_list: list[pd.DataFrame], methods: list[str]) -> None:
    num_vectors = cfg_list[-1].num_vectors
    lr = cfg_list[-1].lr
    lambdaF = cfg_list[-1].zoo.np.lambdaF
    cfg_exec = cfg_list[-1].exec
    print(f'{name} & {header}', end='')
    print(f'{cfg_exec.parameters} & {cfg_exec.resulting_max} & {cfg_exec.blocks} & {num_vectors} & {lr} & {lambdaF} & ', end='')
    data_frame = pd.concat([df.iloc[[-1]] for df in df_list])
    pvalues, data = get_pvalues(data_frame, methods)
    for pv, d in zip(pvalues, data):
        mean = d.mean()
        std = d.std(ddof=1)
        if pv is None:
            print(f'\\textbf{{{mean:.3f}}} $\\pm {std:.3f}$ & ', end='')
        else:
            print(f'{mean:.3f} $\\pm {std:.3f}$ & ', end='')
    print('\b\b\\\\')
    print('pvalues =', pvalues)


def get_pvalues(data_frame: pd.DataFrame, methods: list[str]):
    data = np.array([
        data_frame[(data_frame.method == meth)]['te_acc']
        for meth in methods
    ])
    # identify best mean values
    best_idx = data.mean(axis=1).argmax()
    # p-values by Mann-Whitney U test
    pvalues = [
        # None: best mean value, otherwise p-value
        None if i == best_idx else mannwhitneyu(data[best_idx], data1, alternative='greater')[1]
        for i, data1 in enumerate(data)
    ]
    return pvalues, data


def main():
    epochs = 100
    epochs_cifar10 = 1000
    budget = 1260000
    budget_cifar10 = 100500000
    max_dim = 512

    # read results
    cfg_mnist, df_mnist = read_log.read_dir('mnist', 'MNIST', max_dim, [2, 3, 4], 5e-4, 100, epochs, budget)
    cfg_fmnist, df_fmnist = read_log.read_dir('mnist', 'FMNIST', max_dim, [5, 6, 7], 5e-4, 100, epochs, budget)
    cfg_equalization, df_equalization = read_log.read_dir('equalization', 'equalization', max_dim, [2, 3, 4], 0.005, 10, epochs, budget)
    cfg_copying, df_copying = read_log.read_dir('copying', 'copying', max_dim, [2, 3, 4], 0.0001, 10, epochs, budget)
    cfg_cifar10, df_cifar10 = read_log.read_dir('cifar10', 'CIFAR10', max_dim, [5, 6, 7], 0.001, 100, epochs_cifar10, budget_cifar10)

    # print results
    methods = ['ZO-$\\mathbf{I}$', 'ZO-co', 'ZO-NP']
    header = 'CNN (matrix) & '
    print_line('MNIST', header, cfg_mnist, df_mnist, methods)
    print_line('FashionMNIST', header, cfg_fmnist, df_fmnist, methods)
    header = 'FeedForward (MZI) & '
    print_line('Equalization', header, cfg_equalization, df_equalization, methods)
    header = 'RNN (MZI) & '
    print_line('Copying memory', header, cfg_copying, df_copying, methods)
    header = 'MLP-mixer (matrix) & '
    print_line('CIFAR10', header, cfg_cifar10, df_cifar10, methods)

if __name__ == '__main__':
    main()
