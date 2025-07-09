import os
import hydra
import numpy as np
import torch
import task, train


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{cfg.cuda_id}'
    print('CUDA_VISIBLE_DEVICES = ', os.environ['CUDA_VISIBLE_DEVICES'])
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # dataset and network
    train_loader, test_loader, net = task.get_dataset_model(cfg)
    net.eval()

    # report setting
    outfile = f'/{cfg.suffix}{cfg.dataset}_popsize{cfg.num_vectors}_sigma{cfg.cma_sigma}_{cfg.cuda_id}_{cfg.seed}.log'  
    progr = train.ReportProgress(cfg, outfile)

    # training net by CMA-ES
    tr_cma = train.TrainCMA(cfg, net, progr, train_loader, test_loader)
    try:
        tr_cma.run()
    except KeyboardInterrupt:
        print('KeyboardInterrupt')


if __name__ == '__main__':
    main()
