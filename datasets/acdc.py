import torch
from collections import OrderedDict
import numpy as np
from multiprocessing import Pool

from nnunet.training.dataloading.dataset_loading import load_dataset, DataLoader3D
from nnunet.paths import preprocessing_output_dir
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold
from nnunet.training.data_augmentation.default_data_augmentation import default_3D_augmentation_params, get_default_augmentation

class Plans():

    def __init__(self, plans):
        self.stage = None
        self.process_plans(plans)

    def process_plans(self, plans):
        if self.stage is None:
            assert len(list(plans['plans_per_stage'].keys())) == 1, \
                "If self.stage is None then there can be only one stage in the plans file. That seems to not be the " \
                "case. Please specify which stage of the cascade must be trained"
            self.stage = list(plans['plans_per_stage'].keys())[0]
        self.plans = plans

        stage_plans = self.plans['plans_per_stage'][self.stage]
        # self.batch_size = stage_plans['batch_size']
        self.batch_size = 2
        self.net_pool_per_axis = stage_plans['num_pool_per_axis']
        # self.patch_size = np.array(stage_plans['patch_size']).astype(int)
        self.patch_size = np.array([20,128,112]).astype(int)
        self.do_dummy_2D_aug = stage_plans['do_dummy_2D_data_aug']
        self.net_num_pool_op_kernel_sizes = stage_plans['pool_op_kernel_sizes']
        self.net_conv_kernel_sizes = stage_plans['conv_kernel_sizes']

        self.pad_all_sides = None  # self.patch_size
        self.intensity_properties = plans['dataset_properties']['intensityproperties']
        self.normalization_schemes = plans['normalization_schemes']
        self.base_num_features = plans['base_num_features']
        self.num_input_channels = plans['num_modalities']
        self.num_classes = plans['num_classes'] + 1  # background is no longer in num_classes
        self.classes = plans['all_classes']
        self.use_mask_for_norm = plans['use_mask_for_norm']
        self.only_keep_largest_connected_component = plans['keep_only_largest_region']
        self.min_region_size_per_class = plans['min_region_size_per_class']
        self.min_size_per_class = None  # DONT USE THIS. plans['min_size_per_class']

        if plans.get('transpose_forward') is None or plans.get('transpose_backward') is None:
            print("WARNING! You seem to have data that was preprocessed with a previous version of nnU-Net. "
                  "You should rerun preprocessing. We will proceed and assume that both transpose_foward "
                  "and transpose_backward are [0, 1, 2]. If that is not correct then weird things will happen!")
            plans['transpose_forward'] = [0, 1, 2]
            plans['transpose_backward'] = [0, 1, 2]
        self.transpose_forward = plans['transpose_forward']
        self.transpose_backward = plans['transpose_backward']

        if len(self.patch_size) == 2:
            self.threeD = False
        elif len(self.patch_size) == 3:
            self.threeD = True
        else:
            raise RuntimeError("invalid patch size in plans file: %s" % str(self.patch_size))

        if "conv_per_stage" in plans.keys():  # this ha sbeen added to the plans only recently
            self.conv_per_stage = plans['conv_per_stage']
        else:
            self.conv_per_stage = 2

def get_ACDC_dataset_generator(train_batch_size, val_batch_size):
    t = "Task027_ACDC"
    p = join(preprocessing_output_dir, t, "nnUNetData_plans_v2.1_stage0")
    dataset = load_dataset(p)
    with open(join(join(preprocessing_output_dir, t), "nnUNetPlansv2.1_plans_3D.pkl"), 'rb') as f:
        plans = pickle.load(f)
    # unpack_dataset(p)
    plan = Plans(plans)

    # split data
    splits_file = join(preprocessing_output_dir, "splits_final.pkl")
    if not isfile(splits_file):
        splits = []
        all_keys_sorted = np.sort(list(dataset.keys()))
        kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
        for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            splits.append(OrderedDict())
            splits[-1]['train'] = train_keys
            splits[-1]['val'] = test_keys
        save_pickle(splits, splits_file)

    splits = load_pickle(splits_file)
    # Here we just use fold 0 for training and evaluation.
    tr_keys = splits[0]['train']
    val_keys = splits[0]['val']

    tr_keys.sort()
    val_keys.sort()

    dataset_tr = OrderedDict()
    for i in tr_keys:
        dataset_tr[i] = dataset[i]

    dataset_val = OrderedDict()
    for i in val_keys:
        dataset_val[i] = dataset[i]
    
    # Default patch size is [20, 128, 112].
    dl_tr = DataLoader3D(dataset_tr, plan.patch_size, plan.patch_size, train_batch_size, oversample_foreground_percent=0.33)
    dl_val = DataLoader3D(dataset_val, plan.patch_size, plan.patch_size, val_batch_size,
                         oversample_foreground_percent=0.33)
    # Default train_len=160, val_len=40.
    print(f'training keys length :{len(dl_tr.list_of_keys)}')
    print(f'validation keys length: {len(dl_val.list_of_keys)}')
    tr_gen, val_gen = get_default_augmentation(dl_tr, dl_val,
                                            plan.patch_size,
                                            default_3D_augmentation_params)
    return tr_gen, val_gen

if __name__ == "__main__":

    tr_gen, val_gen = get_ACDC_dataset_generator(4, 4)
    data_dict = next(tr_gen)
    data = data_dict['data']
    target = data_dict['target']

    for i in range(100):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data).float()
        if not isinstance(target, torch.Tensor):
            target = torch.from_numpy(target).float()
        print(f'data:{data.shape}')
        print(f'target:{target.shape}')
        print(f'target max:{target.max()}, target min:{target.min()}')
