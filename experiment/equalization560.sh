#! /bin/bash

python main.py -m seed=1,2,3,4,5 dataset=equalization suffix='equalization560/' cuda_id=2 epochs=100 zoo.max_dim=1024 lr=0.005 zoo.vectors=fromI &
python main.py -m seed=1,2,3,4,5 dataset=equalization suffix='equalization560/' cuda_id=3 epochs=100 zoo.max_dim=1024 lr=0.005 zoo.vectors=coordinate &
python main.py -m seed=1,2,3,4,5 dataset=equalization suffix='equalization560/' cuda_id=4 epochs=100 zoo.max_dim=1024 lr=0.005 zoo.vectors=np zoo.np.lambdaF=10 zoo.budget=1260000 &
