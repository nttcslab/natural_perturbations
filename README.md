# Natural Perturbations for Black-box Training of Neural Networks by Zeroth-Order Optimization

All source code necessary to perform the experiments described in [the above titled manuscript presented at ICML 2025](https://icml.cc/virtual/2025/poster/45125).

## Directories
```
.
├─ compile
├─ experiment
│  ├─ task
│  └─ train
├─ log
│  ├─ block-co-fmnist
│  ├─ cifar10
│  └─ ...
└─ plot
```
- compile: custom CUDA kernels and C++ code for computationally efficient simulation of optical neural networks
- experiment: python code and shell scripts for the experiments
- experiment/task: python code for the models and datasets
- experiment/train: python code for the training methods
- log: to store execution log files in subdirectories, e.g., block-co-fmnist, cifar10.
- plot: python code for generating figures and tables

## Procedure
One can reproduce the experimental results by the following procedure.

1. Prepare a python environment, e.g., Miniconda, and install the following packages (the following command is an install command for Miniconda).
   ```bash
   conda create -n npzoo python=3.11
   conda activate npzoo
   pip install -r requirements.txt
   ``` 
2. Install pytorch by the instructions in https://pytorch.org/get-started/previous-versions/ (the following command is an install command for pytorch 2.6.0 with an environment CUDA 11.8 installed).
   ```bash
   pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu118
   ```
3. (Optional, for the Equalization and Copying memory tasks) Compile CUDA and C++ codes in [compile](./compile/):
   ```bash
   cd compile; pip3 install --no-build-isolation --editable .
   ```
   The built package `mzi_onn_sim` is imported by the python files [experiment/task/chip.py](./experiment/task/chip.py).
4. Run the proposed method (for the MNIST task, by the setting of [config.yaml](./experiment/config.yaml)):
   ```bash
   cd experiment; python main.py
   ```
5. To reproduce all the experimental results, one needs to run all the shell scripts in the directory `experiment`. An example of running a shell script is:
   ```bash
   cd experiment; bash mnist.sh
   ```
   This example shell script [mnist.sh](./experiment/mnist.sh) assumes that the environment has 8 GPUs, `cuda_id` from 0 to 7, and takes advantage of parallel execution.
   When the environment does not have that amount of GPUs, please refer to [none or less GPUs](#none-or-less-gpus).
6. To generate figures and tables, please go to the directory `plot`, and run the corresponding python script. An example is:
   ```bash
   cd plot; python block-co-fmnist.py
   ```

### Files in experiment
#### Python code for Section 3.2
- [entropy_psd_fsd.py](./experiment/entropy_psd_fsd.py): Figure 3 (entropy_psd_fsd.pdf), the relations among the entropy, the expected PSD, and the expected FSD.

#### Python code for running ZO- (Zeroth-Order optimization) and CMA-ES
- [main.py](./experiment/main.py): running ZO-I, ZO-co, ZO-NP
- [main_cma.py](./experiment/main_cma.py): running CME-ES

#### Python code related to the proposed method in `experiment/train`, with comments such as `# line xx, Algorithm 2`
- [train_zo.py](./experiment/train/train_zo.py): ZO optimization

#### Shell script for Experiments
- [mnist.sh](./experiment/mnist.sh): Table 1, Table 2, Figure 12 (MNIST, FashionMNIST)
- [equalization.sh](./experiment/equalization.sh): Table 1, Table 2, Figure 12 (Equalization)
- [copying.sh](./experiment/copying.sh): Table 1, Table 2, Figure 12 (Copying memory)
- [cifar10.sh](./experiment/cifar10.sh): Table 1, Table 2, Figure 6 (CIFAR10)
- [block-co-fmnist.sh](./experiment/block-co-fmnist.sh): Figure 7, Table 3, varying the block size for FashionMNIST
- [variousQ-fmnist.sh](./experiment/variousQ-fmnist.sh): Table 4, varying the number Q of perturbation vectors for FashionMNIST
- [lambdaF.sh](./experiment/lambdaF.sh): Figure 8 (left), varying the FSD hyperparameter
- [Tud.sh](./experiment/Tud.sh): Figure 8 (right), varying the update frequency hyperparameter
- [lr.sh](./experiment/lr.sh): Figure 10, varying the learning rate hyperparameter
- [mnist816.sh](./experiment/mnist816.sh): Figure 11 (MNIST, FashionMNIST)
- [equalization560.sh](./experiment/equalization560.sh): Figure 11 (Equalization)
- [cma.sh](./experiment/cma.sh): Figure 11, performing CMA-ES
- [cifar10large.sh](./experiment/cifar10large.sh): Table 6, Table 7 (CIFAR10 using the enlarged MLP-mixer with N = 1,706,762 parameters)

### Files in plot
- [table1.py](./plot/table1.py): Table 1, overall results
- [table2.py](./plot/table2.py): Table 2, elapsed times and memory footprints
- [conv-time.py](./plot/conv-time.py): Figure 6 [cifar10-conv.pdf](./plot/cifar10-conv.pdf), Figure 12 [others-conv.pdf](./plot/others-conv.pdf), convergence behavior
- [block-co-fmnist.py](./plot/block-co-fmnist.py): Figure 7 [block-co-fmnist.pdf](./plot/block-co-fmnist.pdf), Table 3, FashionMNIST, block coordinate
- [variousQ-fmnist.py](./plot/variousQ-fmnist.py): Table 4, varying the number Q of perturbation vectors for FashionMNIST
- [lambdaF_Tud.py](./plot/lambdaF_Tud.py): Figure 8 [lambdaF_Tud.pdf](./plot/lambdaF_Tud.pdf), varying the FSD hyperparameter and the update frequency hyperparameter
- [lr.py](./plot/lr.py): Figure 10 [lr.pdf](./plot/lr.pdf), varying the learning rate (lr)
- [cma.py](./plot/cma.py): Figure 11 [cma.pdf](./plot/cma.pdf), results with CMA-ES in addition to ZO-I, ZO-co, ZO-NP
- [table6.py](./plot/table6.py): Table 6
- [table7.py](./plot/table7.py): Table 7

## Trouble shooting
### none or less GPUs
If the environment does not have any GPU, i.e., without CUDA, please run a python program as
```bash
python main.py device=cpu
```
If the environment does not have 8 GPUs, please modify a shell script to accommodate the number of GPUs.
For example, if there are 3 GPUs, [mnist.sh](./experiment/mnist.sh) should be modified to
```bash
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=MNIST,FMNIST suffix='mnist/' cuda_id=0 epochs=100 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=MNIST,FMNIST suffix='mnist/' cuda_id=1 epochs=100 zoo.vectors=coordinate &
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=MNIST,FMNIST suffix='mnist/' cuda_id=2 epochs=100 zoo.vectors=np zoo.budget=1260000 &
```
If a shell script is modified and therefore the mapping from `cuda_id` to the experimental condition changes, one should also modify the corresponding plot file in [Files in plot](#files-in-plot).

## Citation

```
@inproceedings{sawada2025natural,
  title = {Natural Perturbations for Black-box Training of Neural Networks by Zeroth-Order Optimization},
  author = {Hiroshi Sawada and Kazuo Aoyama and Yuya Hikima},
  booktitle = {Proceedings of the International Conference on Machine Learning (ICML)},
  year = {2025}
}
```

## License

[SoftwareLicenseAgreement.pdf](./SoftwareLicenseAgreement.pdf)
