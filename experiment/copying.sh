#! /bin/bash

python main.py -m seed=1,2,3,4,5 dataset=copying suffix='copying/' cuda_id=2 epochs=100 lr=1e-4 num_features=32 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5 dataset=copying suffix='copying/' cuda_id=3 epochs=100 lr=1e-4 num_features=32 zoo.vectors=coordinate &
python main.py -m seed=1,2,3,4,5 dataset=copying suffix='copying/' cuda_id=4 epochs=100 lr=1e-4 num_features=32 zoo.vectors=np zoo.np.lambdaF=10 zoo.budget=1260000 &
