#!/bin/bash

# Convert the downloaded ACDC dataset to nnunet detectable.
# Go to datasets folder and run.
python acdc_conversion.py \
--folder_train /afs/crc.nd.edu/user/d/dzeng2/data/acdc/training \
--folder_test /afs/crc.nd.edu/user/d/dzeng2/data/acdc/testing \
--folder_out /afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/

SimpleITK==2.0.2
# Process ACDC dataset.
export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
nnUNet_plan_and_preprocess -t 27 --verify_dataset_integrity

export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
python acdc.py

# train
export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
CUDA_VISIBLE_DEVICES=1 python train.py

# test
export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
CUDA_VISIBLE_DEVICES=2 python test.py \
--checkpoint /afs/crc.nd.edu/user/d/dzeng2/code/icanet/checkpoints/acdc_training_2022-12-04_14-30-11/best.pth \
--test_image_dir /afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/imagesTs \
--test_label_dir /afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/labelsTs \
--output_dir ./result/acdc_unet3d
