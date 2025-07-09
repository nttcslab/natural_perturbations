import pandas as pd
import read_log


def print_line(name: str, df_list: list[pd.DataFrame], methods: list[str]) -> None:
    print(f'{name} & ', end='')
    all_data = pd.concat(df_list)
    last_data = pd.concat([df.iloc[[-1]] for df in df_list])
    for meth in methods:
        ld = last_data[last_data.method == meth]
        time_per_epoch = ld['time'] / ld['epoch']
        print(f'{time_per_epoch.mean():.2f} & ', end='')
        max_memory = all_data[(all_data.method == meth)]['memory'].max()
        print(f'{max_memory:.2f} & ', end='')
    print('\b\b\\\\')


def main():
    epochs = 100
    epochs_cifar10 = 1000
    budget = 1260000
    budget_cifar10 = 100500000
    max_dim = 512

    # read results
    _, df_mnist = read_log.read_dir('mnist', 'MNIST', max_dim, [2, 3, 4], 5e-4, 100, epochs, budget)
    _, df_fmnist = read_log.read_dir('mnist', 'FMNIST', max_dim, [5, 6, 7], 5e-4, 100, epochs, budget)
    _, df_equalization = read_log.read_dir('equalization', 'equalization', max_dim, [2, 3, 4], 0.005, 10, epochs, budget)
    _, df_copying = read_log.read_dir('copying', 'copying', max_dim, [2, 3, 4], 0.0001, 10, epochs, budget)
    _, df_cifar10 = read_log.read_dir('cifar10', 'CIFAR10', max_dim, [5, 6, 7], 0.001, 100, epochs_cifar10, budget_cifar10)

    # print results
    methods = ['ZO-$\\mathbf{I}$', 'ZO-co', 'ZO-NP']
    print_line('MNIST', df_mnist, methods)
    print_line('FashionMNIST', df_fmnist, methods)
    print_line('Equalization', df_equalization, methods)
    print_line('Copying memory', df_copying, methods)
    print_line('CIFAR10', df_cifar10, methods)

if __name__ == '__main__':
    main()
