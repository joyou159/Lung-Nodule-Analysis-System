from collections import namedtuple
import functools
import os
import csv
import numpy as np 
import glob
import SimpleITK as sitk
from util import *
from disk_caching import * 
from torch.utils.data import Dataset 
import torch 
import copy 

@functools.lru_cache(maxsize=1)  # caches the results of function call, if it have been called with the same argument. 
def get_candidate_info_list(dataset_dir_path, required_on_desk=True):
    """ 
    Generate a more sanitized, cleaned, unified interface to the human
    annotated data.
    
    Args:
        required_on_desk (bool): If True, filters candidates based on the subsets present on disk.
    
    Returns:
        list: List of 'CandidateInfoTuple' for all the subjects of existing CT scans.
    """
    mhd_list = glob.glob("/kaggle/input/luna16/subset*/subset*/*.mhd")
    uids_present_on_disk = {os.path.split(p)[-1][:-4] for p in mhd_list} # the unique series_uids for further filtration
    
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
    def __init__(self,dataset_dir_path:str, val_stride:int, val_set_bool:bool = None, series_uid:str = None):
        """
            Initialize training or validation dataset over the entire subjects of specific sujbect 
            by skipping over using a specified validation stride.
            
            Parameters:
                - dataset_dir_path: The path to the dataset directory. 
                - val_stride: The stride used for skipping over nodules candidates to generate training and validation set. 
                    Meaning, if the stride value is '10', this can be interpreted as having training and validation 
                    split of ratio (1/10).
                - val_set_bool: specifies which dataset to return. (training or validation)                 
        """
        
        self.candidates_info_list = copy.copy(get_candidate_info_list(dataset_dir_path)) # to isolate the cached list 
        
        if series_uid: # for specific subject 
            self.candidates_info_list = [x for x in self.candidates_info_list if x.series_uid == series_uid]
        
        if val_set_bool: # return validation set only 
            assert val_stride > 0, val_stride
            self.self.candidates_info_list = self.candidates_info_list[::val_stride]
            assert self.candidates_info_list
        elif val_stride > 0:
            del self.candidates_info_list[::val_stride] # remove this entries, return only the training set 
            assert self.candidates_info_list
            
    
    def __len__(self):
        return len(self.candidate_info_list)
    
    def __getitem__(self, ind):
        candidate_info = self.candidates_info_list[ind] 
        irc_width = IRC_tuple(32, 48, 48) # to make the size of the candidates constant over the training process 
        # ignore the diameter for the sake of unifying the extracted voxel array shape.
        
        curr_ct = CT(candidate_info.series_uid)
        candidate_arr, irc_center = curr_ct.get_raw_candidate_nodule(candidate_info.center_xyz, irc_width)
        
        candidate_tensor = torch.from_numpy(candidate_arr)
        candidate_tensor = candidate_tensor.to(torch.float32)
        candidate_tensor = candidate_tensor.unsqueeze(0) # for the batch-dimension 
        
        label_tensor = torch.tensor([not candidate_info.isNodule_bool, candidate_info.isNodule_bool], dtype = torch.long) 
        
        return (candidate_tensor, label_tensor, candidate_info.series_uid, irc_center) 
        