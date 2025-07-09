# the relations among the entropy, the expected PSD, and the expected FSD.

import os
import hydra
from omegaconf import DictConfig
import numpy as np
import torch
from torch import Tensor
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import shutil
import task, train
from train import ZOoptim


def calc_entropy(covariance: Tensor) -> Tensor:
    nDim = covariance.shape[-1]
    entropy = (nDim * (np.log(2 * np.pi) + 1) + torch.linalg.slogdet(covariance)[1]) / 2
    return entropy


def get_param_entropy(zoo: ZOoptim) -> float:
    entropy: float = 0
    for chol in zoo.chol_matrices:
        inverse = chol @ chol.mT
        Sigma = torch.linalg.inv(inverse)
        entropy += calc_entropy(Sigma).item()
    return entropy


def get_metrics(zoo: ZOoptim) -> tuple[float, float]:
    dout = zoo.dOutputs
    sha = dout.shape
    if len(sha) == 4:  # RNNs
        out_cov = torch.einsum('qbsm,qbsn->bsmn', dout, dout) / sha[0]
    elif len(sha) == 3:  # other NNs
        out_cov = torch.einsum('qbm,qbn->bmn', dout, dout) / sha[0]
    out_ent = calc_entropy(out_cov)
    osd = (dout**2).sum(dim=2).mean().item()
    return out_ent.mean().item(), osd


def get_fsd(zoo: ZOoptim, fisher: Tensor) -> float:
    sha = zoo.dWeights.shape
    start_idx: int = 0
    fsd: float = 0
    for num in zoo.num_params:
        dw = zoo.dWeights[:, start_idx:start_idx+num]
        fsd += torch.einsum('qm,mn,qn->', dw, fisher, dw).item() / sha[0]
        start_idx += num
    return fsd


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.cuda_id}'
    print('CUDA_VISIBLE_DEVICES = ', os.environ['CUDA_VISIBLE_DEVICES'])
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    lambdaF_list = [0, 1, 10]
    lambdaP_min_max = [[1, 5], [1, 5], [1, 5]]
    cfg.dataset = 'MNIST'
    cfg.zoo.max_dim = 1024
    cfg.zoo.vectors = 'fromI'
    cfg.epochs = 5
    cfg.cnn.ch = 3
    cfg.cnn.linear_width = 16

    train_loader, test_loader, net = task.get_dataset_model(cfg)
    net.eval()

    outfile = f'/{cfg.suffix}{cfg.dataset}_{cfg.zoo.vectors}{cfg.zoo.max_dim}_{cfg.cuda_id}_{cfg.seed}.log'
    progr = train.ReportProgress(cfg, outfile)
    tr_inst = train.TrainZO(cfg, net, progr, train_loader, test_loader)

    try:
        tr_inst.train(cfg)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')

    cfg.zoo.vectors = 'np'
    results = pd.DataFrame(columns=['entropy', 'expected PSD', 'output entropy', 'expected OSD',
                                    'expected FSD', 'lambdaP', 'lambdaF', 'iter'])

    num_random = 20
    num_lambdaP = 100
    zoo = ZOoptim(cfg, net)
    for i, (input, label) in enumerate(train_loader):
        _ = zoo.initialize_batch_and_forward(input, label)
        mod_idx, perturbations = zoo.perturb_for_fim()
        zoo.evaluate(perturbations)
        fisher = zoo.calc_fisher()
        for lambdaF, min_max in zip(lambdaF_list, lambdaP_min_max):
            zoo.cfg.np.lambdaF = float(lambdaF)
            lambdaP_list = np.random.uniform(low=min_max[0], high=min_max[1], size=num_lambdaP)
            for lambdaP in lambdaP_list:
                zoo.cfg.np.lambdaP = float(lambdaP)
                zoo.calc_sigma(mod_idx, fisher)
                random = zoo.perturb_normal(num_random)
                zoo.evaluate(random)
                psd = (random**2).sum(dim=1).mean().item()
                ent = get_param_entropy(zoo)
                enty, osd = get_metrics(zoo)
                fsd = get_fsd(zoo, fisher)
                lambdaFstr = f'$\\lambda_F = {lambdaF}$'
                appending = {'entropy': ent, 'expected PSD': psd, 'output entropy': enty,
                             'expected OSD': osd, 'expected FSD': fsd, 'lambdaP': lambdaP,
                             'lambdaF': lambdaFstr, 'iter': i}
                results.loc[len(results)] = pd.Series(appending)
        break

    pdflatex_available = shutil.which("pdflatex") is not None
    plt.rcParams['text.usetex'] = pdflatex_available
    plt.rcParams['pdf.use14corefonts'] = pdflatex_available
    plt.rcParams['font.size'] = 13
    sns.set_style('whitegrid')

    fig, ax = plt.subplots(ncols=2, nrows=1, figsize=(6.5, 3))
    g = sns.scatterplot(data=results, x='expected PSD', y='entropy', hue='lambdaF', style='lambdaF', ax=ax[0], palette="Set1", s=9)
    g.legend(loc='upper left')
    g.set(xscale='log')
    g.set(xticks=[200, 300, 400, 500, 600, 700], xticklabels=['200', '', '', '500', '', ''])
    g.set_xlabel('Expected PSD', fontsize=15)
    g.set_ylabel('Entropy', fontsize=15)
    g = sns.scatterplot(data=results, x='expected FSD', y='entropy', hue='lambdaF', style='lambdaF', ax=ax[1], palette="Set1", s=9)
    g.set(xscale='log', ylabel='')
    g.set(xticks=[10, 100], xticklabels=['10', '100'])
    g.set_xlabel('Expected FSD', fontsize=15)
    g.set_yticks([600, 800, 1000])
    g.set(yticklabels=['', ] * 3)
    g.legend([], [], frameon=False)
    fig.tight_layout()
    plt.savefig('entropy_psd_fsd.pdf', bbox_inches="tight", pad_inches=0)
    plt.show()


if __name__ == '__main__':
    main()
