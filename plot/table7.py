import pandas as pd
import read_log


def read_dir(max_dim_list: list[int]) -> list[pd.DataFrame]:
    dir = 'cifar10large'
    dataset = 'CIFAR10'
    vectors = 'np'
    cuda_ids = [5, 6, 7]
    epochs = 1000
    lr = 0.001
    seed = 1

    df_list = []
    for ci, max_dim in zip(cuda_ids, max_dim_list):
        logfile = f'../log/{dir}/{dataset}_{vectors}{max_dim}_lr{lr}_{ci}_{seed}'
        if vectors == 'np':
            logfile += f'_Tud100_F100'
        logfile += '.log'
        cfg, df = read_log.read_log(logfile, epochs)
        df = df.assign(vectors=vectors)
        df = df.assign(max_dim=cfg.exec.resulting_max)
        df = df.assign(blocks=cfg.exec.blocks)
        df_list.append(df)
    return df_list


def print_line(header: str, all_data: pd.DataFrame, max_dim_list: list[int], epochs: list[int]) -> None:
    for max_dim in max_dim_list:
        print(header, end=' & ')
        ad = all_data[all_data.max_dim==max_dim]
        print(ad['blocks'].iloc[-1], end=' & ')
        for epoch in epochs:
            print(ad[ad.epoch==epoch]['fim_com'].item(), end=' & ')
        print('\b\b\\\\')


def main():
    epochs_cifar10 = 1000
    budget_cifar10 = 120000000
    max_dim = 512
    max_dim_list = [510,]
    _, df_cifar10 = read_log.read_dir('cifar10', 'CIFAR10', max_dim, [5, 6, 7], 0.001, 100, epochs_cifar10, budget_cifar10)
    df_cifar10 = df_cifar10[10]
    header = 'MLP-mixer in Table~\\ref{tab:acc}'
    epochs = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    print_line(header, df_cifar10, max_dim_list, epochs)

    max_dim_list = [512, 256, 128]
    df_cifar10_large = read_dir(max_dim_list)
    df_cifar10_large = pd.concat(df_cifar10_large)
    header = 'Enlarged MLP-mixer'
    epochs = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    print_line(header, df_cifar10_large, max_dim_list, epochs)


if __name__ == '__main__':
    main()
