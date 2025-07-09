#! /bin/bash

python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=MNIST  suffix='mnist/' cuda_id=2 epochs=100 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=MNIST  suffix='mnist/' cuda_id=3 epochs=100 zoo.vectors=coordinate &
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=MNIST  suffix='mnist/' cuda_id=4 epochs=100 zoo.vectors=np zoo.budget=1260000 &
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=FMNIST suffix='mnist/' cuda_id=5 epochs=100 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=FMNIST suffix='mnist/' cuda_id=6 epochs=100 zoo.vectors=coordinate &
python main.py -m seed=1,2,3,4,5 lr=5e-4 dataset=FMNIST suffix='mnist/' cuda_id=7 epochs=100 zoo.vectors=np zoo.budget=1260000 &
