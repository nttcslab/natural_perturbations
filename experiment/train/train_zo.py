''' Zeroth-order Optimization '''
import math
import torch
from torch import Tensor
from omegaconf import DictConfig
from . import util
from task import TensorsLoader


def split_number(N: int, N_max: int) -> list[int]:
    k = -(-N // N_max)  # number of splits
    q = N // k          # quotient
    r = N % k           # residual
    result = [q + 1] * r + [q] * (k - r)
    return result


class TrainZO():
    def __init__(self, cfg: DictConfig, net: torch.nn.Module, progr: util.ReportProgress,
                 train_loader: TensorsLoader, test_loader: TensorsLoader):
        self.num_vectors = cfg.num_vectors
        self.zoo = ZOoptim(cfg, net)
        self.optimizer = getattr(torch.optim, cfg.optimizer)(net.parameters(), lr=cfg.lr)
        progr.report_msg(f'# ZO optimization, #parameters = {self.zoo.nDim}, '
                         f'#blocks = {len(self.zoo.num_params)}, resulting_max = {max(self.zoo.num_params)}')
        msg_append = '\tloss\tgrad_n\ttr_loss\ttr_acc\tte_acc\tn_pertu\tpmodify\tfim_com\tmemory'
        fmt_append = ('{:.5f}', '{:.1e}', '{:.5f}', '{:.5f}', '{:.5f}', '{:d}', '{:d}', '{:d}', '{:.2f}')
        self.progr = progr.set_header(msg_append, fmt_append)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def train(self, cfg) -> None:
        total_iterations = 0
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(cfg.epochs):
            self.progr.start_epoch(epoch, ['loss', 'grad_norm'])
            for (input, label) in self.train_loader:
                loss = self.zoo.initialize_batch_and_forward(input, label)
                match cfg.zoo.vectors:
                    case 'fromI':
                        perturb = self.zoo.perturb_normal(self.num_vectors)
                    case 'coordinate':
                        perturb = self.zoo.perturb_coordinate(self.num_vectors)
                    case 'np':
                        if total_iterations % cfg.zoo.np.Tud == 0:
                            block_idx, perturb1 = self.zoo.perturb_for_fim()  # line 9, Algorithm 2
                            self.zoo.evaluate(perturb1, for_fim=True)         # line 10, Algorithm 2
                            fisher = self.zoo.calc_fisher()                   # line 11-12, Algorithm 2
                            self.zoo.calc_sigma(block_idx, fisher)            # line 13, Algorithm 2
                        perturb = self.zoo.perturb_normal(self.num_vectors)   # line 15-19, Algorithm 2
                    case _:
                        raise ValueError('Unexpected configuration: zoo.vectors')
                self.zoo.evaluate(perturb)
                grad = self.zoo.update_grad()
                self.optimizer.step()
                self.progr.append_results([loss, grad.norm()])
                total_iterations += 1
                if self.zoo.cost_pmodify >= cfg.zoo.budget:
                    break
            _, te_acc = util.calc_loss_accuracy_zo(self.zoo.net, self.test_loader)
            tr_loss, tr_acc = util.calc_loss_accuracy_zo(self.zoo.net, self.train_loader)
            memory_GB = torch.cuda.max_memory_allocated() / 1024**3
            self.progr.progress_report(tr_loss, tr_acc, te_acc, self.zoo.dWeights.shape[0], self.zoo.cost_pmodify, len(self.zoo.block_fim_computed), memory_GB)
            if self.zoo.cost_pmodify >= cfg.zoo.budget:
                return


class ZOoptim():
    def __init__(self, cfg: DictConfig, net):
        self.cfg = cfg.zoo
        self.net = net
        self.device = list(self.net.parameters())[0].device
        self.cost_pmodify = self.cost_forward = int(0)
        self.target_params = []
        self.nDim = 0
        for p in self.net.parameters():
            self.target_params.append(p)
            self.nDim += int(math.prod(p.shape))
        self.num_params = split_number(self.nDim, self.cfg.max_dim)
        self.chol_matrices = []  # the Cholesky decomposition of the inverse of Sigma
        if self.cfg.vectors == 'np':
            for num_param in self.num_params:
                self.chol_matrices.append(torch.eye(num_param, device=self.device))
        elif self.cfg.vectors == 'fromI':
            for num_param in self.num_params:
                self.chol_matrices.append(torch.eye(0, device=self.device))
        self.block_fim_computed: set[int] = set()

    def initialize_batch_and_forward(self, input: Tensor, label: Tensor) -> Tensor:
        self.input = input
        self.label = label
        self.parameters_vector = torch.cat([p.flatten() for p in self.target_params])
        self.output0 = self._chip_forward(self.parameters_vector[None])[0]
        self.loss0 = self.net.criterion(self.output0, self.label)
        return self.loss0

    def perturb_coordinate(self, num_vectors: int) -> Tensor:
        if num_vectors > self.nDim:
            num_vectors = self.nDim
        pert_co = torch.zeros((num_vectors, self.nDim), device=self.device)
        idxs = torch.randperm(self.nDim)[:num_vectors]
        for i, idx in enumerate(idxs):
            pert_co[i][idx] = 1
        return pert_co

    def perturb_normal(self, num_vectors: int) -> Tensor:
        replacement = num_vectors > len(self.num_params)
        bc_list = torch.multinomial(torch.tensor(self.num_params, dtype=torch.float), num_vectors, replacement=replacement)
        randz_list = []
        for i, (num_param, chol) in enumerate(zip(self.num_params, self.chol_matrices)):
            randz = torch.zeros((num_vectors, num_param), device=self.device)
            idx = torch.nonzero(bc_list == i, as_tuple=True)[0]
            len_idx = len(idx)
            if len_idx > 0:
                rand = torch.randn((len_idx, num_param), device=self.device)
                if self.cfg.vectors == 'np':
                    torch.linalg.solve_triangular(chol, rand, upper=False, left=False, out=rand)
                if self.cfg.scale_back is True:
                    rand *= (math.sqrt(num_param) / (rand.norm(dim=1) + torch.finfo(torch.float64).eps))[:, None]  # scale back
                randz[idx] = rand 
            randz_list.append(randz)
        return torch.cat(randz_list, dim=1)

    def perturb_for_fim(self) -> tuple[int, Tensor]:
        block_idx = int(torch.multinomial(torch.tensor(self.num_params, dtype=torch.float), 1))
        self.block_fim_computed.add(block_idx)
        num_perturb = self.num_params[block_idx]
        perturb_list = []
        for i, num_param in enumerate(self.num_params):
            perturb_list.append(
                torch.eye(num_perturb, device=self.device) if i == block_idx
                else torch.zeros((num_perturb, num_param), device=self.device) 
            )
        perturb = torch.cat(perturb_list, dim=1)
        return block_idx, perturb

    def calc_sigma(self, block_idx: int, fisher: Tensor) -> None:
        ''' actually calculate the Cholesky decomposition of the inverse of Sigma '''
        inverse = self.cfg.np.lambdaF * fisher
        ndim = fisher.shape[0]
        indices = torch.arange(ndim)
        inverse[indices, indices] += self.cfg.np.lambdaP
        self.chol_matrices[block_idx][:] = torch.linalg.cholesky(inverse)

    def calc_fisher(self) -> Tensor:
        sha = self.dOutputs.shape
        output_fisher = self.net.calc_output_fisher(self.output0)
        if len(sha) == 4:  # RNNs
            fisher = torch.einsum('pbms,bmks,qbks->pq', self.dOutputs, output_fisher, self.dOutputs) / sha[1] / sha[3]
        elif len(sha) == 3:  # other NNs
            fisher = torch.einsum('pbm,bmk,qbk->pq', self.dOutputs, output_fisher, self.dOutputs) / sha[1]
        else:
            raise ValueError('The shape of self.dOutputs should be in 3 or 4 dimension')
        return fisher

    def evaluate(self, delta: Tensor, for_fim: bool = False) -> None:
        parameters_tensor = self.parameters_vector[None] + self.cfg.mu * delta
        outputs = self._chip_forward(parameters_tensor)
        self.dOutputs = (outputs - self.output0[None]) / self.cfg.mu
        if for_fim is False:
            self.dWeights = delta
            self.dLosses = []
            for i in range(delta.shape[0]):
                loss = self.net.criterion(outputs[i], self.label)
                deltaLoss = (loss - self.loss0) / self.cfg.mu
                self.dLosses.append(deltaLoss)

    def _chip_forward(self, param: Tensor) -> Tensor:
        nVariation, nParams = param.shape
        batch_size = self.input.shape[0]
        self.cost_pmodify += int(nVariation)
        self.cost_forward += int(nVariation * batch_size)
        # expand parameters for nVariation
        params_backup = []
        idx_start = 0
        for p in self.target_params:
            params_backup.append(p.clone())
            psh = p[0].shape
            length = math.prod(psh)
            extracted = param[:, idx_start:idx_start+length]
            extracted = extracted.reshape((nVariation,) + psh)
            p.data = extracted.data
            idx_start += length
        expanded_inputs = torch.stack((self.input, ) * nVariation)
        with torch.no_grad():
            outputs = self.net(expanded_inputs).detach()
        # revert to the original parameters
        for p, b in zip(self.target_params, params_backup):
            p.data = b.data
        return outputs

    def update_grad(self) -> Tensor:
        grad = self._calc_grad_average()
        idx_start = 0
        for p in self.target_params:
            p_shape = p.shape
            length = math.prod(p_shape)
            p.grad = grad[idx_start:idx_start+length].reshape(p_shape)
            idx_start += length
        return grad

    def _calc_grad_average(self) -> Tensor:
        deltaLosses = torch.stack(self.dLosses)
        grad = torch.matmul(deltaLosses, self.dWeights)
        grad /= deltaLosses.shape[0]
        return grad
