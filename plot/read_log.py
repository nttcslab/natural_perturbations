import pandas as pd
import io
import ast
from omegaconf import DictConfig, OmegaConf
import re


def read_log(logfile: str, epochs: int) -> tuple[DictConfig, pd.DataFrame]:
    with open(logfile, 'r') as f:
        lines = f.readlines()
    cfg_list = []
    index_list = []
    for i, line in enumerate(lines):
        if line[0] == '{':
            cfg_dict = ast.literal_eval(line)
            cfg: DictConfig = OmegaConf.create(cfg_dict)
            cfg_list.append(cfg)
        if line[0] == '#':
            index_list.append(i)
    region_start = index_list[-1] + 1
    region_end = region_start + epochs + 1
    cfg = cfg_list[-1]
    df = pd.read_table(io.StringIO(''.join(lines[region_start:region_end])))
    exec_cfg_pairs = re.findall(r'#?(\w+)\s*=\s*(\d+)', lines[region_start-1])
    cfg.update({'exec': {k: int(v) for k, v in exec_cfg_pairs}})
    return cfg, df


def read_dir(dir: str, dataset: str, max_dim: int, cuda_ids: list[int], lr: float,
             lambdaF: int, epochs: int, budget:int) -> tuple[list[DictConfig], list[pd.DataFrame]]:
    vectors_list = ['fromI', 'coordinate', 'np']
    seeds = [1, 2, 3, 4, 5]

    cfg_list = []
    df_list = []
    for ci, vectors in zip(cuda_ids, vectors_list):
        for s in seeds:
            logfile = f'../log/{dir}/{dataset}_{vectors}{max_dim}_lr{lr}_{ci}_{s}'
            if vectors == 'np':
                logfile += f'_Tud100_F{lambdaF}'
            logfile += '.log'
            cfg, df = read_log(logfile, epochs)
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
            df = df.assign(dataset=dataset)
            df = df.assign(max_dim=cfg.exec.resulting_max)
            df = df.assign(blocks=cfg.exec.blocks)
            cfg_list.append(cfg)
            df_list.append(df)
    return cfg_list, df_list
