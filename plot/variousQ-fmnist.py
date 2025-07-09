import numpy as np
import pandas as pd
import read_log
from table1 import get_pvalues


def read_dir(budget_list: list[int]) -> list[pd.DataFrame]:
    dir = 'fmnistQ'
    dataset = 'FMNIST'
    cuda_ids = [1, 2, 3, 4, 5, 6, 7]
    vectors_list = ['fromI', 'coordinate', 'np']
    seeds = [1, 2, 3, 4, 5]
    epochs = 100
    lr = 0.0005
    max_dim = 512

    df_list = []
    for ci, budget in zip(cuda_ids, budget_list):
        for vectors in vectors_list:
            for s in seeds:
                logfile = f'../log/{dir}/{dataset}_{vectors}{max_dim}_lr{lr}_{ci}_{s}'
                if vectors == 'np':
                    logfile += f'_Tud100_F100'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                df = df[df['pmodify'] <= budget]
                match vectors:
                    case 'fromI':
                        method = 'ZO-$\\mathbf{0}$'.format('{I}')
                    case 'coordinate':
                        method = 'ZO-co'
                    case 'np':
                        method = 'ZO-NP'
                df = df.assign(method=method)
                df = df.assign(seed=s)
                df_list.append(df)
    return df_list


def main():
    budget_list = [120000, 180000, 360000, 660000, 1260000, 3060000, 6060000]
    df_list = read_dir(budget_list)
    lastdf = pd.concat([df.iloc[[-1]] for df in df_list])

    Qlist = lastdf['n_pertu'].unique()
    Qlist.sort()
    print('$Q$ & ', end='')
    for q in Qlist:
        print(f'{q} & ', end='')
    print('\b\b\\\\\\hline')

    methods = ['ZO-$\\mathbf{I}$', 'ZO-co', 'ZO-NP']
    pvalues_list = []
    data_list = []
    for q in Qlist:
        lastdf_q = lastdf[lastdf.n_pertu==q]
        pvalues, data = get_pvalues(lastdf_q, methods)
        pvalues_list.append(pvalues)
        data_list.append(data)
    mean_array = np.array(data_list).mean(axis=-1).T
    pvalues_array = np.array(pvalues_list).T
    for meth, pvs, means in zip(methods, pvalues_array, mean_array):
        print(f'{meth} & ', end='')
        for pv, m in zip(pvs, means):
            if pv is None:
                print(f'\\textbf{{{m:.3f}}} & ', end='')
            else:
                print(f'{m:.3f} & ', end='')
        print('\b\b\\\\')
        print('pvalues = [', end='')
        for pv in pvs:
            if pv is None:
                print('None,', end='')
            else:
                print(f'{pv:.5f}, ', end='')
        print(']')

    print('\\hline')
    print('budget ($\\times 10^6$) & ', end='')
    for budget in budget_list:
        budget /= 1e6
        print(f'{budget:.2f} & ', end='')
    print('\b\b\\\\')

    print('ZO-NP epochs & ', end='')
    for q in Qlist:
        epoch = lastdf[(lastdf.method=='ZO-NP') & (lastdf.n_pertu==q)]['epoch'].unique()[0]
        print(f'{epoch} & ', end='')
    print('\b\b\\\\')


if __name__ == '__main__':
    main()
