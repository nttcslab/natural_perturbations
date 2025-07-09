import math
import pickle
import numpy as np
import torch
import cma
from . import util


class TrainCMA():
    def __init__(self, cfg, net, progr, train_loader, test_loader):
        self.net = net
        self.device = cfg.device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_batches_per_epoch = int(train_loader.data_size / cfg.batch_size)
        self.options = {
            'seed': np.random.randint(1e+7),
            'maxiter': cfg.epochs * self.num_batches_per_epoch,
            'popsize': cfg.num_vectors,
            'verbose': -9
        }
        self.savefile = f'/tmp/cma_sigma{cfg.cma_sigma}_{cfg.cuda_id}.pickle'
        self.counter = 0
        self.cost_pmodify = self.cost_forward = int(0)
        self.solution = self._init_param()
        self.sigma = cfg.cma_sigma
        self.es = cma.CMAEvolutionStrategy(self.solution, self.sigma, options=self.options)
        progr.report_msg(f'# CMA-ES, #parameters = {len(self.solution)}')
        msg_append = '\tloss\ttr_loss\ttr_acc\tte_acc\tpopsize\tpmodify'
        fmt_append = ('{:.5f}', '{:.5f}', '{:.5f}', '{:.5f}', '{:d}', '{:d}')
        self.progr = progr.set_header(msg_append, fmt_append)

    def run(self):
        self.train_loader.__iter__()
        self.es.optimize(lambda x: self._fitness_func(x), callback=self._my_callback)
        return self.avg_loss

    def one_epoch(self, epoch):
        if epoch == 0:
            print(self.options)
            self.train_loader.__iter__()
        else:
            print(f'loading from {self.savefile}')
            self.es = pickle.loads(open(self.savefile, 'rb').read())
        self.es.optimize(lambda x: self._fitness_func(x), iterations=self.num_batches_per_epoch, callback=self._my_callback)
        open(self.savefile, 'wb').write(self.es.pickle_dumps())
        return self.avg_loss

    def _my_callback(self, es):
        self.counter = 0
        # es: CMAEvolutionStrategy object
        if es.countiter % self.num_batches_per_epoch == 1:
            epoch = int(es.countiter / self.num_batches_per_epoch)
            self.progr.start_epoch(epoch, ['loss',])
        loss = torch.tensor(self._forward(es.mean))
        self.progr.append_results([loss,])
        if es.countiter % self.num_batches_per_epoch == 0:
            param = torch.tensor(es.mean).to(torch.float32).to(self.device)
            self._set_parameters(param)
            _, te_acc = util.calc_loss_accuracy_zo(self.net, self.test_loader)
            tr_loss, tr_acc = util.calc_loss_accuracy_zo(self.net, self.train_loader)
            self.avg_loss = self.progr.progress_report(tr_loss, tr_acc, te_acc, self.options['popsize'], self.cost_pmodify)

    def _init_param(self):
        param_list = []
        for mod in self.net.modules():
            if len(list(mod.modules())) > 1:
                continue
            for p in mod.parameters():
                param_list.append(p.flatten().detach().cpu())
        return torch.cat(param_list)

    def _set_parameters(self, param):
        nVariation = 1
        idx_start = 0
        for p in self.net.parameters():
            psh = p[0].shape
            length = math.prod(psh)
            extracted = param[idx_start:idx_start+length]
            extracted = extracted.reshape((nVariation,) + psh)
            p.data = extracted.data
            idx_start += length

    def perform_test(self):
        correct_test_list = []
        for i, (input, label) in enumerate(self.test_loader):
            input = self.preprocess(input)
            output_test = self.net(input[None])
            correct_test = self.net.accuracy(output_test[0], label)
            correct_test_list.append(correct_test.item())
        return np.mean(correct_test_list)

    def _get_inputs_label(self):
        try:
            self.inputs, self.label = self.train_loader.__next__()
        except StopIteration:
            self.train_loader.__iter__()
            self.inputs, self.label = self.train_loader.__next__()

    def _forward(self, solution):
        solution = torch.from_numpy(solution).to(torch.float32).to(self.device)
        self._set_parameters(solution)
        expanded_input = torch.cat((self.inputs[None], ))
        output = self.net(expanded_input)
        loss = self.net.criterion(output[0], self.label).item()
        return loss

    def _fitness_func(self, solution):
        if self.counter == 0:
            self._get_inputs_label()
        self.counter += 1
        self.cost_pmodify += 1
        self.cost_forward += self.inputs.shape[0]
        return self._forward(solution)
