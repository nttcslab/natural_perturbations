#! /bin/bash

python main.py -m seed=1,2,3 lr=1e-4,2e-4,5e-4,1e-3,2e-3 dataset=MNIST,FMNIST zoo.vectors=fromI,coordinate,np suffix='lr/' cuda_id=1 zoo.budget=1260000 epochs=100 &
python main.py -m seed=1,2,3 lr=1e-3,2e-3,5e-3,1e-2,2e-2 dataset=equalization zoo.vectors=fromI,coordinate,np suffix='lr/' cuda_id=2 zoo.budget=1260000 epochs=100 zoo.np.lambdaF=10 num_features=16 &
python main.py -m seed=1,2,3 lr=2e-5                     dataset=copying      zoo.vectors=fromI,coordinate,np suffix='lr/' cuda_id=3 zoo.budget=1260000 epochs=100 zoo.np.lambdaF=10 num_features=32 &
python main.py -m seed=1,2,3 lr=5e-5                     dataset=copying      zoo.vectors=fromI,coordinate,np suffix='lr/' cuda_id=4 zoo.budget=1260000 epochs=100 zoo.np.lambdaF=10 num_features=32 &
python main.py -m seed=1,2,3 lr=1e-4                     dataset=copying      zoo.vectors=fromI,coordinate,np suffix='lr/' cuda_id=5 zoo.budget=1260000 epochs=100 zoo.np.lambdaF=10 num_features=32 &
python main.py -m seed=1,2,3 lr=2e-4                     dataset=copying      zoo.vectors=fromI,coordinate,np suffix='lr/' cuda_id=6 zoo.budget=1260000 epochs=100 zoo.np.lambdaF=10 num_features=32 &
python main.py -m seed=1,2,3 lr=5e-4                     dataset=copying      zoo.vectors=fromI,coordinate,np suffix='lr/' cuda_id=7 zoo.budget=1260000 epochs=100 zoo.np.lambdaF=10 num_features=32 &
