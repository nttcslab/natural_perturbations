import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import read_log


def read_dir_lambdaF(pmodify_max: int) -> pd.DataFrame:
    dir = 'lambdaF'
    df_list = []
    cuda_ids = [1, 2, 3, 4]
    dataset_list = ['MNIST', 'equalization', 'copying', 'copying']
    lr_list = [0.0005, 0.005, 0.0001, 0.0001]
    lambdaF_list = [[0, 1, 2, 5, 10, 20, 50, 100, 200], [0, 1, 2, 5, 10, 20, 50, 100], [0, 1, 2, 5], [10, 20, 50, 100, 200]]
    seeds = [1, 2, 3]
    epochs = 100
    max_dim = 512

    for ci, lr, lambdaFs, ds in zip(cuda_ids, lr_list, lambdaF_list, dataset_list):
        for lambdaF in lambdaFs:
            for s in seeds:
                logfile = f'../log/{dir}/{ds}_np{max_dim}_lr{lr}_{ci}_{s}'
                logfile += f'_Tud100_F{lambdaF}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                df = df[df['pmodify'] <= pmodify_max]
                df = df.assign(lambdaF=lambdaF)
                df = df.assign(seed=s)
                match ds:
                    case 'MNIST':
                        dataset = str(ds)
                    case 'equalization':
                        dataset = 'Equalization'
                    case 'copying':
                        dataset = 'Copying memory'
                df = df.assign(dataset=dataset)
                df_list.append(df.iloc[[-1]])
    return pd.concat(df_list)


def read_dir_Tud(pmodify_max: int) -> pd.DataFrame:
    dir = 'Tud'
    df_list = []
    cuda_ids = [1, 2, 3, 4]
    dataset_list = ['MNIST', 'equalization', 'copying', 'copying']
    lr_list = [0.0005, 0.005, 0.0001, 0.0001]
    Tud_list = [[10, 20, 50, 100, 200, 500, 1000], [10, 20, 50, 100, 200, 500, 1000], [10, 20, 50, 100], [200, 500, 1000]]
    lambdaF_list = [100, 10, 10, 10]
    seeds = [1, 2, 3]
    epochs = 100
    max_dim = 512

    for ci, lr, lambdaF, ds, Tuds in zip(cuda_ids, lr_list, lambdaF_list, dataset_list, Tud_list):
        for Tud in Tuds:
            for s in seeds:
                logfile = f'../log/{dir}/{ds}_np{max_dim}_lr{lr}_{ci}_{s}'
                logfile += f'_Tud{Tud}_F{lambdaF}'
                logfile += '.log'
                cfg, df = read_log.read_log(logfile, epochs)
                df = df[df['pmodify'] <= pmodify_max]
                df = df.assign(Tud=Tud)
                df = df.assign(seed=s)
                match ds:
                    case 'MNIST':
                        dataset = str(ds)
                    case 'equalization':
                        dataset = 'Equalization'
                    case 'copying':
                        dataset = 'Copying memory'
                df = df.assign(dataset=dataset)
                df_list.append(df.iloc[[-1]])
    return pd.concat(df_list)


def main():
    pmodify_max = 1260000
    df_lambdaF = read_dir_lambdaF(pmodify_max)
    df_Tud = read_dir_Tud(pmodify_max)

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 13
    sns.set_style('whitegrid')
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.get_cmap("Dark2").colors)

    ylim = [0.89, 1.0]
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(7, 3.5))
    g = sns.pointplot(data=df_lambdaF, x='lambdaF', y='te_acc', hue='dataset', markers=['o', 's', '+'], linestyles=['-', '--', ':'], ax=ax[0], errorbar='sd', dodge=True)
    g.set(ylabel='', title='', ylim=ylim)
    g.set_xlabel(r'FSD weight $\lambda_F$', fontsize=14)
    g.legend([],[], frameon=False)
    g = sns.pointplot(data=df_Tud, x='Tud', y='te_acc', hue='dataset', markers=['o', 's', '+'], linestyles=['-', '--', ':'], ax=ax[1], errorbar='sd', dodge=True)
    g.set(title='', ylabel='', ylim=ylim)
    g.set_xlabel(r'Update frequency $T_\mathrm{ud}$', fontsize=14)
    g.legend(loc='lower left')
    fig.tight_layout()
    plt.savefig('lambdaF_Tud.pdf', bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
