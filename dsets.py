from collections import namedtuple
import functools
import os
import csv
import numpy as np 
import glob
import SimpleITK as sitk
from util import *
from disk_caching import * 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset 
import copy 
import random


@functools.lru_cache(maxsize=1)  # caches the results of function call, if it have been called with the same argument. 
def get_candidate_info_list(dataset_dir_path, required_on_desk=True, subsets_included = (0,1,2,3,4)):
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
    filtered_mhd_list = [mhd for mhd in mhd_list if any(f"subset{id}" in mhd for id in subsets_included)]
    uids_present_on_disk = {os.path.split(p)[-1][:-4] for p in filtered_mhd_list} # the unique series_uids for further filtration
    
    annotation_info = dict()
    
    with open(os.path.join(dataset_dir_path, "annotations.csv")) as f:
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
            
            if series_uid not in uids_present_on_disk and required_on_desk: 
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
        ct_chunks, center_irc = get_ct_raw_candidates(series_uid, center_xyz, width_irc) 
    else:
        ct = get_ct(series_uid)
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


class CT: 
    def __init__(self, series_uid):
        mhd_path = glob.glob(f"/kaggle/input/luna16/subset*/subset*/{series_uid}.mhd")[0] # since returns a list 
        ct_mhd = sitk.ReadImage(mhd_path) # this method consumes the .raw file implicitly  
        ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype = np.float32).clip(-1000, 1000)
        self.series_uid = series_uid
        self.hu_arr = ct_arr
        self.origin_xyz = XYZ_tuple(*ct_mhd.GetOrigin())
        self.voxel_sizes = XYZ_tuple(*ct_mhd.GetSpacing())
        self.transform_mat = np.array(ct_mhd.GetDirection()).reshape(3, 3)

    def get_raw_candidate_nodule(self, xyz_center, irc_diameters):
        """
        This method extracts the nodule and a small amount of data from its surrounding 
        for limiting the scope of the problem for the classifier.  
        
        Parameters: 
            - xyz_center (XYZ_tuple): the center of the potential nodule expressed in the patient space.  
            - irc_diameters (IRC_tuple): the spread of the nodule around its center expressed in the voxel space.
                                   This constant and don't depend on the diameter of the nodule for consistency.  
        
        return: 
            - ct_chunk: the tiny volumentric surrounding of the suspicious nodule. 
            - icr_center: The transformed center coordinates. 
        """
        icr_center = xyz2irc(xyz_center, self.origin_xyz, self.voxel_sizes, self.transform_mat) 
        
        slicing_list = list()
        for i, i_center in enumerate(icr_center):
            start_ind = int(round(i_center - irc_diameters[i]/2))
            end_ind = int(start_ind + irc_diameters[i])
            
            # report if there is any issue 
            assert i_center >= 0 and i_center < self.hu_arr.shape[i], repr([self.series_uid, xyz_center, self.origin_xyz, self.voxel_sizes, icr_center, i])
            
            # safety checks 
            if start_ind < 0: 
                start_ind = 0 
                end_ind = int(irc_diameters[i])
            
            if end_ind > self.hu_arr.shape[i]:
                end_ind = self.hu_arr.shape[i]
                start_ind = int(end_ind - irc_diameters[i])
            

            slicing_list.append(slice(start_ind, end_ind))
        
        ct_chunks = self.hu_arr[tuple(slicing_list)]
        
        return ct_chunks, icr_center
    

@functools.lru_cache(maxsize = 1, typed = True) # this would be enough if we are sure that the nodules loading will occur in order 
# meaning that all the candidate nodules of specific subject is first extracted, then the second subject's nodules and so on. 
def get_ct(series_uid):
    return CT(series_uid)

raw_cache = getCache("cache_candidates")

@raw_cache.memoize(typed = True) # save on disk to avoid loading the same ct scan each time to extract specific nodule surrounding 
# (just upload it once and save it for further nodules extraction from the same subject)
def get_ct_raw_candidates(series_uid ,xyz_center, irc_diameters):
    ct = get_ct(series_uid)
    ct_chunks, icr_center = ct.get_raw_candidate_nodule(xyz_center, irc_diameters)
    return ct_chunks, icr_center



class LunaDataset(Dataset):
    def __init__(self, dataset_dir_path:str,
                subsets_included:tuple = (0,1,2,3,4) ,
                val_stride:int = 0,
                val_set_bool:bool = None, 
                ratio_int:int = 0 ,
                series_uid:str = None,
                sortby_str:str='random',
                augmentation_dict = None):
        """
            Initialize training or validation dataset over the entire subjects of specific sujbect 
            by skipping over using a specified validation stride.
            
            Parameters:
                - dataset_dir_path: The path to the dataset directory.
                - subsets_included: The subsets of the dataset to be included. 
                - val_stride: The stride used for skipping over nodules candidates to generate training and validation set. 
                    Meaning, if the stride value is '10', this can be interpreted as having training and validation 
                    split of ratio (1/10).
                - val_set_bool: specifies which dataset to return. (training or validation)
                - ratio_int: The ratio between negative and postive samples in the dataset, for the sake of balancing.
                - series_uid: extract the nodule candidates of specific subject.
                - sortby_str: the ordering criteria used, among ('random', 'series_uid' and 'label_and_size' (default)).            
        """
        self.ratio_int = ratio_int 
        self.augmentation_dict = augmentation_dict
        self.candidates_info_list = copy.copy(get_candidate_info_list(DATASET_DIR_PATH, required_on_desk=True,subsets_included = subsets_included)) # to isolate the cached list 
        
        if series_uid: # for specific subject 
            self.candidates_info_list = [x for x in self.candidates_info_list if x.series_uid == series_uid] 

        if val_set_bool: # return validation set only 
            assert val_stride > 0, val_stride
            self.candidates_info_list = self.candidates_info_list[::val_stride]
            assert self.candidates_info_list
        elif val_stride > 0:
            del self.candidates_info_list[::val_stride] # remove this entries, return only the training set 
            assert self.candidates_info_list
    
        if sortby_str == 'random':
            random.shuffle(self.candidates_info_list)
        elif sortby_str == 'series_uid':
            self.candidates_info_list.sort(key=lambda x: (x.series_uid, tuple(x.center_xyz)))
        elif sortby_str == 'label_and_size': # the default
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))
        

        # separation of the positive and negative samples for the sake of balancing.
        self.positive_list = [cand for cand in self.candidates_info_list if cand.isNodule_bool]
        self.negative_list = [cand for cand in self.candidates_info_list if not cand.isNodule_bool]

        log.info(f'{repr(self)}: {len(self.candidates_info_list)} {"validation" if val_set_bool else "training"} samples, {len(self.negative_list)} neg, {len(self.positive_list)} pos, {self.ratio_int if self.ratio_int else 'unbalanced'} ratio')
   
    def __len__(self): 
        if self.ratio_int:
            candidates_count = len(self.candidates_info_list)  # at least  
            inbetwee_count = len(self.negative_list) // self.ratio_int 
            if inbetwee_count >= len(self.positive_list):
                candidates_count += (inbetwee_count - len(self.positive_list))
            return candidates_count 
        else:
            return len(self.candidates_info_list) 
    
    def __getitem__(self, ind):
        # implementing alternating mechanism to balance the dataset
        if self.ratio_int:
            pos_ind = ind // (self.ratio_int + 1)   
            if ind % (self.ratio_int + 1): # a non-zero reminder means this should be a negative sample
                neg_ind = ind - 1 - pos_ind
                neg_ind %= len(self.negative_list) # overflow results in wraparound
                candidate_info = self.negative_list[neg_ind]
            else:
                pos_ind %= len(self.positive_list) # overflow results in wraparound
                candidate_info = self.positive_list[pos_ind]            
        else:
            candidate_info = self.candidates_info_list[ind] 


        irc_width = IRC_tuple(32, 48, 48) # to make the size of the candidates constant over the training process 
        # ignore the diameter for the sake of unifying the extracted voxel array shape.

        if self.augmentation_dict:
            candidate_tensor, irc_center = get_ct_augmented_candidates(self.augmentation_dict, candidate_info.series_uid, candidate_info.center_xyz, irc_width)
        else: 
            candidate_arr, irc_center = get_ct_raw_candidates(candidate_info.series_uid ,candidate_info.center_xyz, irc_width)
            candidate_tensor = torch.from_numpy(candidate_arr)
            candidate_tensor = candidate_tensor.to(torch.float32)
            candidate_tensor = candidate_tensor.unsqueeze(0) # for the batch-dimension 
            
    
        label_tensor = torch.tensor([not candidate_info.isNodule_bool, candidate_info.isNodule_bool], dtype = torch.long) 
        
        return (candidate_tensor, label_tensor, candidate_info.series_uid, irc_center) 
        
    # custom shuffling for each epoch (you can neglect and use the pytorch optional argument)
    def shuffle_samples(self):
        if self.ratio_int:
            random.shuffle(self.negative_list)
            random.shuffle(self.positive_list)
        




