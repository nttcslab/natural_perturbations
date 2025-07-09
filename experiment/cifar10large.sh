#! /bin/bash

python main.py seed=1 dataset=CIFAR10 suffix='cifar10large/' cuda_id=3 epochs=1000 num_vectors=200 mlp.n_mixer=12 mlp.n_channel=256 zoo.vectors=fromI              save_file='../log/cifar10large/fromI.pt' &
python main.py seed=1 dataset=CIFAR10 suffix='cifar10large/' cuda_id=4 epochs=1000 num_vectors=200 mlp.n_mixer=12 mlp.n_channel=256 zoo.vectors=coordinate         save_file='../log/cifar10large/coordinate.pt' &
python main.py seed=1 dataset=CIFAR10 suffix='cifar10large/' cuda_id=5 epochs=1000 num_vectors=200 mlp.n_mixer=12 mlp.n_channel=256 zoo.vectors=np                 save_file='../log/cifar10large/np.pt' &
python main.py seed=1 dataset=CIFAR10 suffix='cifar10large/' cuda_id=6 epochs=1000 num_vectors=200 mlp.n_mixer=12 mlp.n_channel=256 zoo.vectors=np zoo.max_dim=256 save_file='../log/cifar10large/np256.pt' &
python main.py seed=1 dataset=CIFAR10 suffix='cifar10large/' cuda_id=7 epochs=1000 num_vectors=200 mlp.n_mixer=12 mlp.n_channel=256 zoo.vectors=np zoo.max_dim=128 save_file='../log/cifar10large/np128.pt' &
