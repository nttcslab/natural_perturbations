#! /bin/bash

python main.py -m seed=1,2,3,4,5 dataset=CIFAR10 suffix='cifar10/' cuda_id=5 epochs=1000 num_vectors=200 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5 dataset=CIFAR10 suffix='cifar10/' cuda_id=6 epochs=1000 num_vectors=200 zoo.vectors=coordinate &
python main.py -m seed=1,2,3,4,5 dataset=CIFAR10 suffix='cifar10/' cuda_id=7 epochs=1000 num_vectors=200 zoo.vectors=np zoo.budget=100500000 &
