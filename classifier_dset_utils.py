from collections import namedtuple
import torch 
import random
import functools
from common_utils.util import *
from common_utils.disk_caching import * 
import os
import csv
import numpy as np 
import glob
import torch.nn as nn 
from CT import get_ct, get_ct_raw_candidates



@functools.lru_cache(maxsize=1)  # caches the results of function call, if it have been called with the same argument. 
def get_unified_candidate_info_list(dataset_dir_path, required_on_desk=True, subsets_included = (0,1,2,3,4)):
    """ 
    Generate a more sanitized, cleaned, unified interface to the human
    annotated data.
    
    Args:
        dataset_dir_path (str): The path to dataset dir. 
        required_on_desk (bool): If True, filters candidates based on the subsets present on disk.
        num_subsets_included (tuple): the subsets to be included from the on-desk dataset. 
                                    This is specifically useful in case of computational limitation. 
    Returns:
        list: List of 'CandidateInfoTuple' for all the subjects of existing CT scans.
    """
    
    mhd_list = glob.glob("/kaggle/input/luna16/subset*/subset*/*.mhd") # extract all 
    if required_on_desk:
        filtered_mhd_list = [mhd for mhd in mhd_list if any(f"subset{id}" in mhd for id in subsets_included)]
        uids_present_on_disk = {os.path.split(p)[-1][:-4] for p in filtered_mhd_list} # the unique series_uids for further filtration
        mhd_list = uids_present_on_disk

    annotation_info = dict()
    
    with open(os.path.join(dataset_dir_path, "annotations.csv"), "r") as f:
        for row in list(csv.reader(f))[1:]: # neglect the first indexing column 
            series_uid = row[0]
            nodule_center = np.array([float(x) for x in row[1:4]]) # x, y, z 
            nodule_diameter = float(row[4])
            annotation_info.setdefault(series_uid, []).append((nodule_center, nodule_diameter)) 
            # meaning that same series_uid has multiple annotations for mutliple nodules  
    
    candidate_list = list()
    with open(os.path.join(dataset_dir_path, "candidates.csv")) as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            
            if series_uid not in mhd_list:
                continue # meaning that subject doesn't exist on the desk 
                
            nodule_flag = bool(int(row[4])) # is it an actual nodule?
            cand_nodule_center = np.array([float(x) for x in row[1:4]])
            cand_nodule_diameter = 0.0 # the assigned value to the fake nodules (incorrect assumption but won't hurt)
            
            for annotation_tup in annotation_info.get(series_uid, []):  # this way of iterating provides me a default value (robust to missing key error)
                nodule_center, nodule_diameter = annotation_tup 
                diff = np.abs(cand_nodule_center - nodule_center)
                # check whether there is any huge (more that half the raduis) deviation in any direction 
                if not (np.all(diff < nodule_diameter / 4)):  
                    pass # leave the candidate diameter zero, as it's actually a fake candidate  
                else:
                    cand_nodule_diameter = nodule_diameter 
                    break # meaning i reached to the annotation corresponding to what i need, so get out. 
            candidate_list.append(
                CandidateInfoTuple(nodule_flag,
                                   cand_nodule_diameter,
                                   series_uid,
                                   cand_nodule_center))
            
    candidate_list.sort(key = lambda x: (x.isNodule_bool, x.diameter_mm),reverse=True) # sort by 'nodule_flag' first, then by 'cand_nodule_diameter' 
    # so overall, the returned list has the candidates that are nodules indeed ordered descendingly by their sizes  
    # after that come the non-nodule candidates
    return candidate_list



@functools.lru_cache(maxsize=1)  
def get_ct_augmented_candidates(augmentation_dict, series_uid, center_xyz, width_irc, use_cache = True):
    if use_cache:
        ct_chunks, center_irc = get_ct_raw_candidates(series_uid, center_xyz, width_irc, usage ="classifier") 
    else:
        ct = get_ct(series_uid, usage ="classifier")
        ct_chunks, center_irc = ct.get_raw_candidate_nodule(center_xyz, width_irc)

    ct_tensor = torch.tensor(ct_chunks).unsqueeze(0).unsqueeze(0).to(torch.float32) # the (batch_size, channel, depth, heigth, width)  the expected shape of pytorch 

    transformation_mat = torch.eye(4) # start off (just identity)

    for i in range(3): # per axis 
        if augmentation_dict['flip']:
            if random.random() > 0.5: # flipping is a bit random 
                transformation_mat[i:i] *= -1

        if augmentation_dict['offset']:
            offset_value = augmentation_dict['offset'] # must be limited [-1,1]
            random_factor = random.random() * 2 - 1  # (std -> 2) and (mean -> -1)
            transformation_mat[i:4] = offset_value * random_factor # the grid_sample will interpolate since the transition won't be in voxel steps 

        if augmentation_dict['scale']:
            scaling_value = augmentation_dict["scale"]
            random_factor = (random.random() * 2 - 1)
            transformation_mat[i:i] *= 1 + scaling_value * random_factor
    
    # rotation around z-axis (because the scale of this axis is completely different from those of x & y)
    if augmentation_dict['rotate']:
        angle_in_rad = random.random() * np.pi * 2
        s = np.sin(angle_in_rad)
        c = np.cos(angle_in_rad)

        rotation_mat = torch.tensor(
            [
                [c, -s, 0, 0],
                [s, c, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ]
        ) 
                             
        transformation_mat @= rotation_mat # accumulate the transformation matrices

    affine_transform = nn.functional.affine_grid(transformation_mat[:3].unsqueeze(0).to(torch.float32), # transformation_mat[:3] coordinates only (not density value) 
                                                 ct_tensor.size, align_corners=False) 
    
    augmented_chunk = nn.functional.grid_sample(ct_tensor, affine_transform, padding_mode="border", align_corners=False).to('cpu') 

    
    if augmentation_dict['noise' ]:
        noise_t = torch.randn_like(augmented_chunk)
        noise_t *= augmentation_dict['noise']
        augmented_chunk += noise_t

    
    return augmented_chunk[0], center_irc # (discard the batch dim, just up to the channel dim)
