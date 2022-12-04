#!/bin/bash

#$ -M dzeng2@nd.edu	 # Email address for job notification
#$ -m abe		 # Send mail when job begins, ends and aborts
#$ -q gpu
#$ -l gpu_card=1
#$ -N ssl_train        # Specify job name

conda activate py3.8

export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
python train.py --epochs 200 --lr 1e-4