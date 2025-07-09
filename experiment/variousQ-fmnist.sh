#! /bin/bash

python main.py -m seed=1,2,3,4,5 zoo.vectors=fromI,coordinate,np cuda_id=1 num_vectors=1   lr=5e-4 dataset=FMNIST suffix='fmnistQ/' epochs=100 zoo.budget=120000 &
python main.py -m seed=1,2,3,4,5 zoo.vectors=fromI,coordinate,np cuda_id=2 num_vectors=2   lr=5e-4 dataset=FMNIST suffix='fmnistQ/' epochs=100 zoo.budget=180000 &
python main.py -m seed=1,2,3,4,5 zoo.vectors=fromI,coordinate,np cuda_id=3 num_vectors=5   lr=5e-4 dataset=FMNIST suffix='fmnistQ/' epochs=100 zoo.budget=360000 &
python main.py -m seed=1,2,3,4,5 zoo.vectors=fromI,coordinate,np cuda_id=4 num_vectors=10  lr=5e-4 dataset=FMNIST suffix='fmnistQ/' epochs=100 zoo.budget=660000 &
python main.py -m seed=1,2,3,4,5 zoo.vectors=fromI,coordinate,np cuda_id=5 num_vectors=20  lr=5e-4 dataset=FMNIST suffix='fmnistQ/' epochs=100 zoo.budget=1260000 &
python main.py -m seed=1,2,3,4,5 zoo.vectors=fromI,coordinate,np cuda_id=6 num_vectors=50  lr=5e-4 dataset=FMNIST suffix='fmnistQ/' epochs=100 zoo.budget=3060000 &
python main.py -m seed=1,2,3,4,5 zoo.vectors=fromI,coordinate,np cuda_id=7 num_vectors=100 lr=5e-4 dataset=FMNIST suffix='fmnistQ/' epochs=100 zoo.budget=6060000 &
