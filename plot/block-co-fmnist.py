import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import read_log


def read_dir(budget: int) -> list[pd.DataFrame]:
    dir = 'block-co-fmnist'
    dataset = 'FMNIST'
    cuda_ids = [2, 3, 4, 1, 5, 6, 7]
    vectors_list = ['fromI', 'fromI', 'fromI', 'coordinate', 'np', 'np', 'np']
    max_dim_list = [[1,512], [2,8,128], [4,16,32,64,256], [1,], [1,512], [2,8,128], [4,16,32,64,256]]
    max_resulting_max = 431
    seeds = [1, 2, 3, 4, 5]
    epochs = 100
    lr = 0.0005

    df_list = []
    for ci, vectors, max_dims in zip(cuda_ids, vectors_list, max_dim_list):
        for max_dim in max_dims:
            for s in seeds:
                logfile = f'../log/{dir}/{dataset}_{vectors}{max_dim}_lr{lr}_{ci}_{s}'
                if vectors == 'np':
                    logfile += f'_Tud100_F100'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                df = df[df['pmodify'] <= budget]
                df = df.assign(max_dim=cfg.exec.resulting_max)
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
                if vectors == 'coordinate':
                    se = df.iloc[[-1]].copy()
                    se['max_dim'] = max_resulting_max
                    df_list.append(se)
    return df_list


def report_memory_footprint(df_list):
    all_data = pd.concat(df_list)
    zonp_data = all_data[(all_data.method == 'ZO-NP')]
    max_dim_list = zonp_data['max_dim'].unique()
    max_dim_list.sort()
    max_dim_list = max_dim_list[2:]
    print('$N_\\mathrm{max}$ & ', end='')
    for max_dim in max_dim_list:
        print(f'{max_dim} & ', end='')
    print('\b\b\\\\\\hline')
    print('memory & ', end='')
    for max_dim in max_dim_list:
        max_memory = zonp_data[zonp_data['max_dim']==max_dim]['memory'].max()
        print(f'{max_memory:.2f} & ', end='')
    print('\b\b\\\\')


def main():
    budget = 1260000
    df_list = read_dir(budget)
    report_memory_footprint(df_list)

    lastdf = pd.concat([df.iloc[[-1]] for df in df_list])

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 14
    sns.set_style('whitegrid')

    ylim = [0.783, 0.863]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6, 3.5))
    g = sns.lineplot(ax=ax, data=lastdf, x='max_dim', y='te_acc', hue='method', style='method', errorbar='sd')
    g.set(xscale='log', xlim=[1, 431], ylabel='', ylim=ylim)
    g.set_xlabel('$N_\\mathrm{max}$', fontsize=16)
    g.set_ylabel('Test accuracy', fontsize=16)
    g.set(yticks=[0.80, 0.82, 0.84, 0.86])
    g.legend()
    fig.tight_layout()
    # plt.savefig('block-co-fmnist.png', bbox_inches="tight", dpi=500)
    plt.savefig('block-co-fmnist.pdf', bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
