#! /bin/bash

python main.py -m seed=1,2,3,4,5 dataset=FMNIST lr=5e-4 suffix='block-co-fmnist/' cuda_id=1 epochs=100 zoo.vectors=coordinate zoo.max_dim=1 &
python main.py -m seed=1,2,3,4,5 dataset=FMNIST lr=5e-4 suffix='block-co-fmnist/' cuda_id=2 epochs=100 zoo.vectors=fromI zoo.max_dim=1,512          zoo.scale_back=true &
python main.py -m seed=1,2,3,4,5 dataset=FMNIST lr=5e-4 suffix='block-co-fmnist/' cuda_id=3 epochs=100 zoo.vectors=fromI zoo.max_dim=2,8,128        zoo.scale_back=true &
python main.py -m seed=1,2,3,4,5 dataset=FMNIST lr=5e-4 suffix='block-co-fmnist/' cuda_id=4 epochs=100 zoo.vectors=fromI zoo.max_dim=4,16,32,64,256 zoo.scale_back=true &
python main.py -m seed=1,2,3,4,5 dataset=FMNIST lr=5e-4 suffix='block-co-fmnist/' cuda_id=5 epochs=100 zoo.vectors=np    zoo.max_dim=1,512          zoo.scale_back=true zoo.budget=1260000 &
python main.py -m seed=1,2,3,4,5 dataset=FMNIST lr=5e-4 suffix='block-co-fmnist/' cuda_id=6 epochs=100 zoo.vectors=np    zoo.max_dim=2,8,128        zoo.scale_back=true zoo.budget=1260000 &
python main.py -m seed=1,2,3,4,5 dataset=FMNIST lr=5e-4 suffix='block-co-fmnist/' cuda_id=7 epochs=100 zoo.vectors=np    zoo.max_dim=4,16,32,64,256 zoo.scale_back=true zoo.budget=1260000 &
