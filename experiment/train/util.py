import time
import datetime
import numpy as np
import pandas as pd
from task import TensorsLoader


def calc_loss_accuracy_zo(net, data_loader: TensorsLoader) -> tuple[float, float]:
    loss_list, correct_list = [], []
    for i, (input, label) in enumerate(data_loader):
        output = net(input[None])
        loss = net.criterion(output[0], label)
        loss_list.append(loss.item())
        correct = net.accuracy(output[0], label)
        correct_list.append(correct)
    return float(np.mean(loss_list)), float(np.mean(correct_list))


# def calc_loss_accuracy_bp(net, data_loader: TensorsLoader) -> tuple[float, float]:
#     loss_list, correct_list = [], []
#     for i, (input, label) in enumerate(data_loader):
#         output = net(input)
#         loss = net.criterion(output, label)
#         loss_list.append(loss.item())
#         correct = net.accuracy(output, label)
#         correct_list.append(correct)
#     return float(np.mean(loss_list)), float(np.mean(correct_list))


class ReportProgress():
    def __init__(self, cfg, outfile):
        self.start_time = time.time()
        self.outfile = cfg.logdir + outfile
        dt = datetime.datetime.fromtimestamp(self.start_time)
        self._print(dt)
        self._print(cfg)

    def set_header(self, msg_append, fmt_append):
        msg = 'epoch\ttime'
        msg += msg_append
        self.fmt = ('{:d}', '{:.2f}')
        self.fmt += fmt_append
        self._print(msg)
        return self

    def start_epoch(self, epoch, columns):
        self.epoch = epoch
        self.results = pd.DataFrame(columns=columns)

    def append_results(self, results):
        appending = {k: v.item() for k, v in zip(self.results.columns, results)}
        app_se = pd.Series(appending)
        self.results.loc[len(self.results)] = app_se

    def progress_report(self, *direct_values):
        elapsed_time = time.time() - self.start_time
        mean_results = self.results.mean()
        to_report = [self.epoch+1, elapsed_time]
        to_report.extend([v for v in mean_results])
        to_report.extend(direct_values)
        msg = ''
        for v, f in zip(to_report, self.fmt):
            msg += f.format(v) + '\t'
        self._print(msg[:-1])
        loss_mean = self.results['loss'].mean()
        return loss_mean

    def report_msg(self, msg):
        self._print(msg)

    def _print(self, msg):
        print(msg)
        if self.outfile is not None:
            with open(self.outfile, 'a') as f:
                print(msg, file=f)
