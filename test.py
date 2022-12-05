import os
import argparse
import pandas as pd
import torch.nn as nn
from models.RecursiveUNet3D import UNet3D
from nnunet.inference.predict import *
from nnunet.paths import preprocessing_output_dir
from nnunet.preprocessing.preprocessing import GenericPreprocessor

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str, default='../checkpoints/acdc_training_2022-12-04_12-55-01/best.pth', help="checkpoint path")
parser.add_argument('--test_image_dir', type=str, default='/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/imagesTs', help="test image path")
parser.add_argument('--test_label_dir', type=str, default='/afs/crc.nd.edu/user/d/dzeng2/data/acdc/converted/nnUNet_raw_data/Task027_ACDC/labelsTs', help="test label path")
parser.add_argument('--output_dir', type=str, default='./result/acdc_unet3d', help="where is the output saved.")

def preprocess_patient(input_files, plans):
    preprocessor = GenericPreprocessor(plans['normalization_schemes'], plans['use_mask_for_norm'],
                                       plans['transpose_forward'],
                                       plans['dataset_properties']['intensityproperties'])
    d, s, properties = preprocessor.preprocess_test_case(input_files, plans['plans_per_stage'][0]['current_spacing'])
    return d, s, properties


def main():
    args = parser.parse_args()
    # set up cuda
    torch.cuda.set_device('cuda:0')

    # load model
    model = UNet3D(num_classes=4, in_channels=1, initial_filter_size=64, kernel_size=3, num_downs=4, norm_layer=nn.InstanceNorm3d)

    # load model parameters
    model.load_state_dict(torch.load(args.checkpoint)['state_dict'])
    model.cuda()
    model.eval()

    # specify the folder that contain all the test images
    input_folder = args.test_image_dir
    # folder that you hope the result will be saved
    output_folder = args.output_dir
    maybe_mkdir_p(output_folder)
    # give the plan file while you train your model
    with open(join(join(preprocessing_output_dir, 'Task027_ACDC'), "nnUNetPlansv2.1_plans_3D.pkl"), 'rb') as f:
        plans = pickle.load(f)

    expected_num_modalities = plans['num_modalities']
    case_ids = check_input_folder_and_return_caseIDs(input_folder, expected_num_modalities)
    output_files = [join(output_folder, i + ".nii.gz") for i in case_ids]
    all_files = subfiles(input_folder, suffix=".nii.gz", join=False, sort=True)
    list_of_images = [[join(input_folder, i) for i in all_files if i[:len(j)].startswith(j) and
                    len(i) == (len(j) + 12)] for j in case_ids]
    list_of_labels = [join(args.test_label_dir, i + ".nii.gz") for i in case_ids]

    save_npz = False
    torch.cuda.empty_cache()

    for i, l in enumerate(list_of_images):
        output_file = output_files[i]
        print("preprocessing", output_file)
        data, _, dct = preprocess_patient(l, plans)
        print("predicting", output_file)
        softmax = model.predict_3D(data, do_mirroring=False, use_sliding_window=True, patch_size=[20,128,112], use_gaussian=True, all_in_gpu=True, verbose=True)[1][None]
        transpose_forward = plans.get('transpose_forward')
        if transpose_forward is not None:
            transpose_backward = plans['transpose_backward']
            # print(f'transpose_backward:{transpose_backward}')
            softmax = np.mean(softmax, 0)
            # print(f'softmax:{softmax.shape}')
            softmax = softmax.transpose([0] + [i + 1 for i in transpose_backward])
        if save_npz:
            npz_file = output_file[:-7] + ".npz"
        else:
            npz_file = None
        softmax = softmax.astype(np.float)
        save_segmentation_nifti_from_softmax(softmax, output_file, dct, 3, None, None, None,
                                                npz_file, None, None, 0)

    # Compute dice coefficent.
    columns = ['case_ids', 'dice']
    rows = []
    dice_list = []
    for i, output_file in enumerate(output_files):
        label_file = list_of_labels[i]
        # Read prediction and label.
        prediction = sitk.ReadImage(output_file, sitk.sitkUInt8)
        label = sitk.ReadImage(label_file, sitk.sitkUInt8)
        overlap_measures_filter = sitk.LabelOverlapMeasuresImageFilter()
        overlap_measures_filter.Execute(label, prediction)
        dice = overlap_measures_filter.GetDiceCoefficient()
        print(f'{case_ids[i]}, dice:{dice:.3f}')
        rows.append([case_ids[i], dice])
        dice_list.append(dice)
    print(f'Average dice:{np.array(dice_list).mean()}')
    # Uncomment this if you want to save the individual result to excel.
    # results_df = pd.DataFrame(rows, columns=columns)
    # results_df.to_excel(os.path.join('./result/', 'dice_results.xlsx'), index=False)

if __name__ == "__main__":
    main()
