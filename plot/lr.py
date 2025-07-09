import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import read_log


def read_dir(pmodify_max: int) -> pd.DataFrame:
    dir = 'lr'
    df_list = []
    cuda_ids = [1, 2, 3, 4, 5, 6, 7]
    dataset_list = [['MNIST', 'FMNIST'], ['equalization',], ['copying',], ['copying',], ['copying',], ['copying',], ['copying',]]
    lr_list = [[1e-4,2e-4,5e-4,1e-3,2e-3], [1e-3,2e-3,5e-3,1e-2,2e-2], [2e-5,], [5e-5,], [1e-4,], [2e-4,], [5e-4]]
    seeds = [1, 2, 3]
    vector_list = ['fromI', 'coordinate', 'np']
    lambdaF_list = [100, 10, 10, 10, 10, 10, 10]
    epochs = 100
    max_dim = 512

    for ci, lrs, lambdaF, datasets in zip(cuda_ids, lr_list, lambdaF_list, dataset_list):
        for ds in datasets:
            for vector in vector_list:
                for lr in lrs:
                    for s in seeds:
                        logfile = f'../log/{dir}/{ds}_{vector}{max_dim}_lr{lr}_{ci}_{s}'
                        if vector == 'np':
                            logfile += f'_Tud100_F{lambdaF}'
                        logfile += '.log'
                        cfg, df = read_log.read_log(logfile, epochs)
                        df = df[df['pmodify'] <= pmodify_max]
                        df = df.assign(lr=lr)
                        df = df.assign(seed=s)
                        match vector:
                            case 'fromI':
                                method = 'ZO-$\\mathbf{0}$'.format('{I}')
                            case 'coordinate':
                                method = 'ZO-co'
                            case 'np':
                                method = 'ZO-NP'
                        match cfg.dataset:
                            case 'MNIST':
                                dataset = str(cfg.dataset)
                            case 'FMNIST':
                                dataset = str(cfg.dataset)
                            case 'equalization':
                                dataset = 'Equalization'
                            case 'copying':
                                dataset = 'Copying memory'
                        df = df.assign(method=method)
                        df = df.assign(dataset=dataset)
                        df_list.append(df.iloc[[-1]])
    return pd.concat(df_list)


def main():
    pmodify_max = 1260000
    lastdf = read_dir(pmodify_max)

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 13
    sns.set_style('whitegrid')
    sns.color_palette("colorblind")

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(14, 3.5))
    g = sns.pointplot(data=lastdf[lastdf.dataset=='MNIST'], x='lr', y='te_acc', hue='method', markers=['o', 's', '+'], linestyles=['-', '--', ':'], ax=ax[0], errorbar='sd', dodge=True)
    g.set(title='MNIST')
    g.set_ylabel('Test accuracy', fontsize=15)
    g.set_xlabel('Learning rate (lr)', fontsize=15)
    g.legend()
    g = sns.pointplot(data=lastdf[lastdf.dataset=='FMNIST'], x='lr', y='te_acc', hue='method', markers=['o', 's', '+'], linestyles=['-', '--', ':'], ax=ax[1], errorbar='sd', dodge=True)
    g.set(title='FashionMNIST', ylabel='')
    g.set_xlabel('Learning rate (lr)', fontsize=15)
    g.set(yticks=[0.75, 0.80, 0.85])
    g.legend([],[], frameon=False)
    g = sns.pointplot(data=lastdf[lastdf.dataset=='Equalization'], x='lr', y='te_acc', hue='method', markers=['o', 's', '+'], linestyles=['-', '--', ':'], ax=ax[2], errorbar='sd', dodge=True)
    g.set(title='Equalization', ylabel='')
    g.set_xlabel('Learning rate (lr)', fontsize=15)
    g.legend([],[], frameon=False)
    g = sns.pointplot(data=lastdf[lastdf.dataset=='Copying memory'], x='lr', y='te_acc', hue='method', markers=['o', 's', '+'], linestyles=['-', '--', ':'], ax=ax[3], errorbar='sd', dodge=True)
    g.set(title='Copying memory', ylabel='')
    g.set_xlabel('Learning rate (lr)', fontsize=15)
    g.set(yticks=[0.90, 0.95, 1.00])
    g.legend([],[], frameon=False)
    fig.tight_layout()
    plt.savefig('lr.pdf', bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
