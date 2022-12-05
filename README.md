### Step1: Download ACDC dataset from this [link](http://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb).


### Step2: The data preprocessing part is heavily relied on [nnunet](https://github.com/MIC-DKFZ/nnUNet), so we need to convert the original data to nnunet compatible for processing.

Go to the datasets/ folder and run (please change the paths to the path on your own machine).
```
python acdc_conversion.py \
--folder_train /afs/crc.nd.edu/user/d/dzeng2/data/acdc/training \
--folder_test /afs/crc.nd.edu/user/d/dzeng2/data/acdc/testing \
--folder_out /afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/
```
Please keep the `nnUNet_raw_data/Task027_ACDC/` in `--folder_out` because it's important for nnunet to find the path.

### Step3: Install nnunet and run preprocessing.

To install nnunet
```
git clone https://github.com/MIC-DKFZ/nnUNet.git
cd nnUNet
pip install -e .
```

To run preprocessing
```
export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
nnUNet_plan_and_preprocess -t 27 --verify_dataset_integrity
```
`nnUNet_raw_data_base` is the base folder that you save the acdc conversion file (remove the `nnUNet_raw_data/Task027_ACDC/` part here)
`nnUNet_preprocessed` is where you want to preprocessed file to be saved.

### Step4: Model training.

To train a model
Go to the icanet/ folder (base folder) and run
```
export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
CUDA_VISIBLE_DEVICES=0 python train.py
```
`nnUNet_raw_data_base` and `nnUNet_preprocessed` are the same as Step3.


### Step5: Test the model accuracy.

We use 3d-sliding window to predict multiple patches of a sample and fuse them to generate the final prediction.
```
export nnUNet_raw_data_base="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted"; \
export nnUNet_preprocessed="/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_preprocessed"; \
CUDA_VISIBLE_DEVICES=0 python test.py \
--checkpoint /afs/crc.nd.edu/user/d/dzeng2/code/icanet/checkpoints/acdc_training_2022-12-04_16-38-48/best.pth \
--test_image_dir /afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/imagesTs \
--test_label_dir /afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/labelsTs \
--output_dir ./result/acdc_unet3d
```
`test_image_dir` and `test_label_dir` should be in the `folder_out` folder specified in Step1. Prediction results will be saved in the `output_dir` folder

### Pretrained model

Here is a pretrained Unet3D model [link](https://drive.google.com/drive/folders/1ujIVnpkufIlcuoJXMqtEnIWxHoH7xiMH?usp=sharing).
