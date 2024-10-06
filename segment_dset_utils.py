import functools
import glob
import os
import csv
import torch.nn as nn
import torch 
from collections import namedtuple
import random
from math import sin, cos, pi


CandidateSegInfo = namedtuple('CandidateSegInfo', 'is_nodule, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')

@functools.lru_cache(maxsize=1)  # caches the results of function call, if it have been called with the same argument. 
def get_candidate_info_list(dataset_dir_path, required_on_desk=True, subsets_included = (0,1,2,3,4)):

    mhd_list = glob.glob("/kaggle/input/luna16/subset*/subset*/*.mhd") # extract all 
    if required_on_desk:
        filtered_mhd_list = [mhd for mhd in mhd_list if any(f"subset{id}" in mhd for id in subsets_included)]
        uids_present_on_disk = {os.path.split(p)[-1][:-4] for p in filtered_mhd_list} # the unique series_uids for further filtration
        mhd_list = uids_present_on_disk

    candidates_list = list()

    # extract the nodules (whether they are malignant or benign) that has annotations data without repetition
    with open(os.path.join(dataset_dir_path, "annotations_for_segmentation.csv"), "r") as f:
        for row in list(csv.reader(f))[1:]: 
            series_uid = row[0]
            if series_uid not in mhd_list:
                continue

            center_xyz = tuple([float(x) for x in row[1:4]])
            diameter = float(row[4])
            is_malignant = {"False": False, "True": True}[row[5]] 


            candidates_list.append(
                CandidateSegInfo(
                    True, # it's a nodule 
                    True, # has annotations data already (meaning that there are some nodules that have no annotation data)
                    is_malignant,
                    diameter,
                    series_uid, 
                    center_xyz
                )
            )
    
    # extract non-nodule (negative examples so that the U-net model learns to ignore them)
    with open(os.path.join(dataset_dir_path, "annotations.csv"), "r") as f:
        for row in list(csv.reader(f))[1:]: 
            series_uid = row[0]

            if series_uid not in mhd_list:
                continue

            is_nodule = bool(int(row[4]))
            center_xyz = tuple([float(x) for x in row[1:4]])

            if not is_nodule: 
                candidates_list.append(
                    CandidateSegInfo(
                        False,   
                        False,
                        False,
                        diameter,
                        series_uid, 
                        center_xyz
                    )
                )

    return candidates_list


def find_radius(ci, cr, cc, axis, hu_arr, threshold_hu):
    """
    For mask generation 
    """
    radius = 2
    try:
        while True:
            # Check based on the axis
            if axis == 'index':
                if hu_arr[ci + radius, cr, cc] <= threshold_hu or hu_arr[ci - radius, cr, cc] <= threshold_hu:
                    break
            elif axis == 'row':
                if hu_arr[ci, cr + radius, cc] <= threshold_hu or hu_arr[ci, cr - radius, cc] <= threshold_hu:
                    break
            elif axis == 'col':
                if hu_arr[ci, cr, cc + radius] <= threshold_hu or hu_arr[ci, cr, cc - radius] <= threshold_hu:
                    break
            # Increment the radius if the condition is met
            radius += 1
    except IndexError:
        radius -= 1  # Fix the last incorrect incrementation due to out-of-bounds access
    return radius

@functools.lru_cache(1)
def get_candidate_info_dict(dataset_dir_path, required_on_desk=True, subsets_included = (0,)):
    candidate_list = get_candidate_info_list(dataset_dir_path, required_on_desk, subsets_included)
    candidate_dict = dict()

    for candidate_tuple in candidate_list:
        candidate_dict.setdefault(candidate_tuple.series_uid, []).append(candidate_tuple)
    
    return candidate_dict


"""
The main purpose of this class is to make the preprocess step of data augmentation to be done on the GPUs.
"""
class SegmentationAugmentation(nn.Module):
    def __init__(self, flip=None, offset=None, scale=None, rotate=None, noise=None):
        super().__init__()

        self.flip = flip 
        self.offset = offset 
        self.scale = scale 
        self.rotate = rotate
        self.noise = noise 


    def forward(self, input_batch, mask_batch):
        """
        input_batch: batch of size (7, height, width)
        mask_batch: batch of size (1, height, width)

        """
        transform_mat = self.build_2d_transformation_matrix()
        transform_mat = transform_mat.expand(input_batch.shape[0], -1, -1) # account for batch dimension 
        transform_mat = transform_mat.to(input_batch.device, torch.float32) # loading the matrix to GPU
        
        transform_grid  = nn.functional.affine_grid(transform_mat[:,:2], # when defining the grid, we just take the first two rows
                                                      input_batch.size(), align_corners=False) 

        augmented_input = nn.functional.grid_sample(input_batch, 
                                                    transform_grid, 
                                                    padding_mode="border", 
                                                    align_corners=False)
        
        augmented_masks = nn.functional.grid_sample(mask_batch.to(torch.float32),  # because by default it's (torch.long), inconvenient    
                                                    transform_grid, 
                                                    padding_mode="border", 
                                                    align_corners=False)

        if self.noise:
            noise_added = torch.rand_like(augmented_input) * self.noise
            augmented_input += noise_added

        return augmented_input, augmented_masks > 0.5 # convert back to boolean. 

    def build_2d_transformation_matrix(self):
        transform_mat = torch.eye(3) 

        for i in range(2):
            if self.flip:
                if random.random() > 0.5: # flipping is a bit random 
                    transform_mat[i:i] *= -1

            if self.offset:
                offset_value = self.offset # must be limited [-1,1]
                random_factor = (random.random() * 2 - 1)  # (std -> 2) and (mean -> -1)
                transform_mat[i,2] = offset_value * random_factor # the grid_sample will interpolate since the transition won't be in voxel steps 

            if self.scale:
                scaling_value = self.scale
                random_factor = (random.random() * 2 - 1)
                transform_mat[i:i] *= 1 + scaling_value * random_factor        

        if self.rotate:
            rotation_angle = random.random() * pi * 2
            s = sin(rotation_angle)
            c = cos(rotation_angle)

            rotation_matrix = torch.tensor([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])

            transform_mat @= rotation_matrix
    
        return transform_mat

    