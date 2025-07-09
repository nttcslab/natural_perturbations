#! /bin/bash

python main.py -m seed=1,2,3 zoo.np.lambdaF=0,1,2,5,10,20,50,100,200 dataset=MNIST        lr=0.0005 suffix='lambdaF/' cuda_id=1 epochs=100 zoo.vectors=np zoo.budget=1260000 &
python main.py -m seed=1,2,3 zoo.np.lambdaF=0,1,2,5,10,20,50,100     dataset=equalization lr=0.005  suffix='lambdaF/' cuda_id=2 epochs=100 zoo.vectors=np zoo.budget=1260000 &
python main.py -m seed=1,2,3 zoo.np.lambdaF=0,1,2,5                  dataset=copying      lr=0.0001 suffix='lambdaF/' cuda_id=3 epochs=100 zoo.vectors=np zoo.budget=1260000 num_features=32 &
python main.py -m seed=1,2,3 zoo.np.lambdaF=10,20,50,100,200         dataset=copying      lr=0.0001 suffix='lambdaF/' cuda_id=4 epochs=100 zoo.vectors=np zoo.budget=1260000 num_features=32 &
