#! /bin/bash

python main_cma.py -m seed=1,2,3,4,5 cma_sigma=0.005 dataset=MNIST  cnn.ch=3 epochs=100 num_vectors=21 cuda_id=5 suffix='cma/' &
python main_cma.py -m seed=1,2,3,4,5 cma_sigma=0.002 dataset=FMNIST cnn.ch=3 epochs=100 num_vectors=21 cuda_id=6 suffix='cma/' &
python main_cma.py -m seed=1,2,3,4,5 cma_sigma=0.009 dataset=equalization    epochs=100 num_vectors=21 cuda_id=7 suffix='cma/' &
