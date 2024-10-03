import torch
from torch.utils.data import Dataset
from CT import getCtSampleSize, get_ct
from logconfig import *
from segment_dset_utils import get_candidate_info_dict, get_candidate_info_list

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

class Luna2DSegmentation(Dataset):
    def __init__(self, dataset_dir_path:str,
                subsets_included:tuple = (0,1,2,3,4) ,val_stride = 0, val_set_bool = None, series_uid= None, context_slices = 3, full_ct=True):
        super(Luna2DSegmentation, self).__init__()
        """
        Base class for  training or validation dataset over the entire subjects of specific sujbect 
        by skipping over using a specified validation stride.
        
        Parameters:
            - dataset_dir_path: The path to the dataset directory.
            - subsets_included: The subsets of the dataset to be included. 
            - val_stride: The stride used for skipping over subjects ct to generate training and validation set. 
                Meaning, if the stride value is '10', this can be interpreted as having training and validation 
                split of ratio (1/10).
            - val_set_bool: specifies which dataset to return. (training or validation)
            - series_uid: extract the nodule candidates of specific subject.
            - context_slice: the number of slices above and below the current slice at the current slice taken into account 
                during segmentation. 
            - full_ct: divide based on just positive indicies in the ct volume or the whole volume.
        """
        self.context_slices = context_slices
        self.full_ct = full_ct

        if series_uid:
            self.subjects_list = [series_uid]
        else:
            self.subjects_list = sorted(get_candidate_info_dict(dataset_dir_path, subsets_included = subsets_included, 
                                                                required_on_desk=True).keys())
            
        if val_set_bool: # return validation set only 
            assert val_stride > 0, val_stride
            self.subjects_list = self.subjects_list[::val_stride]
            assert self.subjects_list
        elif val_stride > 0:
            del self.subjects_list[::val_stride] # remove this entries, return only the training set 
            assert self.subjects_list 
        
        self.sample_list = list()
        for series_uid in self.subjects_list:
            num_indicies, positive_slices = getCtSampleSize(series_uid)

            if self.full_ct: # get the whole volume slices
                self.sample_list += [(series_uid, slice_ind) 
                                     for slice_ind in range(num_indicies)]  
            else:
                self.sample_list += [(series_uid, slice_ind) 
                                     for slice_ind in positive_slices]
        
        # uploading individual candidates from each subject included in the subject list 
        self.candidate_list = get_candidate_info_list(dataset_dir_path, True, subsets_included)
        self.candidate_list = [sub for sub in self.candidate_list if sub.series_uid in self.subjects_list]

        self.pos_list = [sub for sub in self.candidate_list if sub.is_nodule] 

        log.info("{!r}: {} {} series, {} slices {} nodules".format(self, len(self.subjects_list), 
                                             {None: "general", True: "validation", False: "training"}[val_set_bool],
                                             len(self.sample_list),
                                             len(self.pos_list)))

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        series_uid, slice_ind = self.sample_list[index % len(self.sample_list)] # circulation 
        return self.slice_extract_with_context(series_uid, slice_ind)
    
    def slice_extract_with_context(self, series_uid, slice_ind):
        ct = get_ct(series_uid, usage = "segment")
        ct_t = torch.zeros((self.context_slices * 2 + 1, 512, 512)) # taking self.context_slices above and below the target slice for context learning.

        start_ind = slice_ind - self.context_slices
        end_ind = slice_ind + self.context_slices + 1
        for i, context_idx in enumerate(range(start_ind, end_ind)):
            context_idx = max(context_idx, 0)
            context_idx = min(context_idx, ct.hu_arr.shape[0] - 1)
            ct_t[i] = torch.from_numpy(ct.hu_arr[context_idx])
        
        labels = torch.from_numpy(ct.positive_masks[slice_ind]).unsqueeze(0) # for batching
        return ct_t, labels, series_uid, slice_ind
    
