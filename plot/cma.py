import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import shutil
import read_log


def read_dir_cma(dataset: str, cuda_ids: list, sigmas_list: list[list], seeds: list) -> pd.DataFrame:
    dir = 'cma'
    lastdf = pd.DataFrame()
    epochs = 100

    for ci, sigmas in zip(cuda_ids, sigmas_list):
        for sigma in sigmas:
            for s in seeds:
                logfile = f'../log/{dir}/{dataset}_popsize21_sigma{sigma}_{ci}_{s}.log'
                cfg, df = read_log.read_log(logfile, epochs)
                df = df.assign(method='CMA-ES')
                df = df.assign(seed=s)
                df = df.assign(sigma=sigma)
                df = df.assign(dataset=cfg.dataset)
                lastdf = pd.concat([lastdf, df.iloc[[-1]]])
    return lastdf


def main():
    epochs = 100
    budget = 1260000

    _, df_mnist = read_log.read_dir('mnist816', 'MNIST', 1024, [2, 3, 4], 0.0005, 100, epochs, budget)
    df_mnist = pd.concat([df.iloc[[-1]] for df in df_mnist])
    _, df_fmnist = read_log.read_dir('mnist816', 'FMNIST', 1024, [5, 6, 7], 0.0005, 100, epochs, budget)
    df_fmnist = pd.concat([df.iloc[[-1]] for df in df_fmnist])
    _, df_equalization = read_log.read_dir('equalization560', 'equalization', 1024, [2, 3, 4], 0.005, 10, epochs, budget)
    df_equalization = pd.concat([df.iloc[[-1]] for df in df_equalization])
    cma_mnist = read_dir_cma('MNIST', [5,], [[0.005,],], [1,2,3,4,5])
    cma_fmnist = read_dir_cma('FMNIST', [6,], [[0.002,],], [1,2,3,4,5])
    cma_equalization = read_dir_cma('equalization', [7,], [[0.009,],], [1,2,3,4,5])
    df_mnist = pd.concat([df_mnist, cma_mnist[cma_mnist.sigma==0.005]])
    df_fmnist = pd.concat([df_fmnist, cma_fmnist[cma_fmnist.sigma==0.002]])
    df_equalization = pd.concat([df_equalization, cma_equalization[cma_equalization.sigma==0.009]])

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 13
    sns.set_style('whitegrid')
    sns.color_palette("colorblind")

    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 3))
    g = sns.boxplot(ax=ax[0], data=df_mnist, y='te_acc', hue='method')
    g.set_title('MNIST', fontsize=15)
    g.set_ylabel('Test accuracy', fontsize=15)
    g.legend([],[], frameon=False)
    g = sns.boxplot(ax=ax[1], data=df_fmnist, y='te_acc', hue='method')
    g.set_title('FashionMNIST', fontsize=15)
    g.set_ylabel('')
    g.legend([],[], frameon=False)
    g = sns.boxplot(ax=ax[2], data=df_equalization, y='te_acc', hue='method')
    g.set_title('Equalization', fontsize=15)
    g.set_ylabel('')
    g.set_ylim([0.865, 1.00])
    g.set(yticks=[0.88, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    g.legend(bbox_to_anchor=(1, 1), loc='upper left')
    fig.tight_layout()
    plt.savefig('cma.pdf', bbox_inches="tight", pad_inches=0)
    plt.savefig('cma.png', bbox_inches="tight", dpi=500)
    plt.show()


if __name__ == '__main__':
    main()
