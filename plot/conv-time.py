import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import read_log


seed = 2


def create_figure(df_list: list[pd.DataFrame], name: str, ylim_train: list[float], ylim_test: list[float]) -> None:
    dataframe = pd.concat(df_list)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
    g = sns.lineplot(data=dataframe[dataframe.seed==seed], x='epoch', y='loss', hue='method', style='method', ax=ax[0])
    g.set(ylabel='')
    g.set_xlabel('Epoch', fontsize=15)
    if ylim_train is not None:
        g.set_ylim(ylim_train)
    g.set_title(r'Training loss ($\downarrow$)', fontsize=15)
    g.legend()
    g = sns.lineplot(data=dataframe[dataframe.seed==seed], x='time', y='te_acc', hue='method', style='method', ax=ax[1])
    g.set(ylabel='')
    g.set_xlabel('Elapsed time (seconds)', fontsize=15)
    if ylim_train is not None:
        g.set_ylim(ylim_test)
    g.set_title(r'Test accuracy ($\uparrow$)', fontsize=15)
    g.legend()
    fig.tight_layout()
    plt.savefig(f'{name}-conv.pdf', bbox_inches="tight", pad_inches=0)
    plt.show()


def create_figure_for_others(
        df_list_list: list[list[pd.DataFrame]],
        names: list[str],
        ylims_train: list[list[float]],
        ylims_test: list[list[float]]
    ) -> None:
    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(14, 7))
    for i, (df_list, name, ylim_train, ylim_test) in enumerate(zip(df_list_list, names, ylims_train, ylims_test)):
        dataframe = pd.concat(df_list)
        g = sns.lineplot(data=dataframe[dataframe.seed==seed], x='epoch', y='loss', hue='method', style='method', ax=ax[0, i])
        if i == 0:
            g.set_ylabel('Training loss', fontsize=15)
        else:
            g.set(ylabel='')
        g.set_xlabel('Epoch', fontsize=15)
        if ylim_train is not None:
            g.set_ylim(ylim_train)
        g.set_title(name, fontsize=15)
        g.legend()
        g = sns.lineplot(data=dataframe[dataframe.seed==seed], x='time', y='te_acc', hue='method', style='method', ax=ax[1, i])
        if i == 0:
            g.set_ylabel('Test accuracy', fontsize=15)
        else:
            g.set(ylabel='')
        g.set_xlabel('Elapsed time (seconds)', fontsize=15)
        if ylim_train is not None:
            g.set_ylim(ylim_test)
        g.legend()
    fig.tight_layout()
    plt.savefig(f'others-conv.pdf', bbox_inches="tight", pad_inches=0)
    plt.show()


def main():
    epochs = 100
    epochs_cifar10 = 1000
    budget = 1260000
    budget_cifar10 = 100500000
    max_dim = 512

    _, df_mnist = read_log.read_dir('mnist', 'MNIST', max_dim, [2, 3, 4], 5e-4, 100, epochs, budget)
    _, df_fmnist = read_log.read_dir('mnist', 'FMNIST', max_dim, [5, 6, 7], 5e-4, 100, epochs, budget)
    _, df_equalization = read_log.read_dir('equalization', 'equalization', max_dim, [2, 3, 4], 0.005, 10, epochs, budget)
    _, df_copying = read_log.read_dir('copying', 'copying', max_dim, [2, 3, 4], 0.0001, 10, epochs, budget)
    _, df_cifar10 = read_log.read_dir('cifar10', 'CIFAR10', max_dim, [5, 6, 7], 0.001, 100, epochs_cifar10, budget_cifar10)
    
    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 13
    sns.set_style('whitegrid')

    create_figure(df_cifar10, 'cifar10', [0.78, 1.5], [0.5, 0.63])
    create_figure_for_others(
        [df_mnist, df_fmnist, df_equalization, df_copying],
        ['MNIST', 'FashionMNIST', 'Equalization', 'Copying memory'],
        [[0.05, 0.45], [0.35, 0.88], [0, 0.9], [0, 0.9]],
        [[0.89, 0.985], [0.67, 0.87], [0.78, 1.0], [0.78, 1.0]]
    )


if __name__ == '__main__':
    main()
