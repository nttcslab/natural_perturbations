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
    if cfg.load_file is not None:
        net.load_state_dict(torch.load(cfg.load_file))
    net.eval()

    # report setting
    outfile = f'/{cfg.suffix}{cfg.dataset}_{cfg.zoo.vectors}{cfg.zoo.max_dim}_lr{cfg.lr}_{cfg.cuda_id}_{cfg.seed}'
    if cfg.zoo.vectors == 'np':
        outfile += f'_Tud{cfg.zoo.np.Tud}_F{cfg.zoo.np.lambdaF}'
    outfile += '.log'
    progr = train.ReportProgress(cfg, outfile)

    # training net by ZO optimization
    tr_zoo = train.TrainZO(cfg, net, progr, train_loader, test_loader)
    try:
        tr_zoo.train(cfg)
    except KeyboardInterrupt:
        print('KeyboardInterrupt')
    if cfg.save_file is not None:
        torch.save(net.state_dict(), cfg.save_file)

if __name__ == '__main__':
    main()
