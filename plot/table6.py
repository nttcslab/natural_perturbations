import pandas as pd
import read_log


def read_dir(max_dim_list: list[int]) -> list[pd.DataFrame]:
    dir = 'cifar10large'
    dataset = 'CIFAR10'
    cuda_ids = [3, 4, 5, 6, 7]
    vectors_list = ['fromI', 'coordinate', 'np', 'np', 'np']
    epochs = 340
    lr = 0.001
    seed = 1

    df_list = []
    for ci, vectors, max_dim in zip(cuda_ids, vectors_list, max_dim_list):
        logfile = f'../log/{dir}/{dataset}_{vectors}{max_dim}_lr{lr}_{ci}_{seed}'
        if vectors == 'np':
            logfile += f'_Tud100_F100'
        logfile += '.log'
        cfg, df = read_log.read_log(logfile, epochs)
        match vectors:
            case 'fromI':
                method = 'ZO-$\\mathbf{0}$'.format('{I}')
            case 'coordinate':
                method = 'ZO-co'
            case 'np':
                method = 'ZO-NP'
        df = df.assign(method=method)
        df = df.assign(max_dim=cfg.exec.resulting_max)
        df = df.assign(blocks=cfg.exec.blocks)
        df_list.append(df)
    return df_list


def print_line(df_list: list[pd.DataFrame], methods: list[str], max_dim_list: list[int]) -> None:
    all_data = pd.concat(df_list)
    last_data = pd.concat([df.iloc[[-1]] for df in df_list])
    for meth, max_dim in zip(methods, max_dim_list):
        print(f'{meth} & ', end='')
        ld = last_data[(last_data.method==meth) & (last_data.max_dim==max_dim)]
        print(ld['max_dim'].item(), end=' & ')
        print(ld['blocks'].item(), end=' & ')
        time_per_epoch = ld['time'] / ld['epoch']
        print(f'{time_per_epoch.mean():.2f} & ', end='')
        max_memory = all_data[(all_data.method==meth) & (all_data.max_dim==max_dim)]['memory'].max()
        print(f'{max_memory:.2f} & ', end='')
        print('\b\b\\\\')


def main():
    # read results
    max_dim_list = [512, 512, 512, 256, 128]
    df_cifar10 = read_dir(max_dim_list)

    # print results
    methods = ['ZO-$\\mathbf{I}$', 'ZO-co', 'ZO-NP', 'ZO-NP', 'ZO-NP']
    print_line(df_cifar10, methods, max_dim_list)

if __name__ == '__main__':
    main()
