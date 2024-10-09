import torch
from torch.utils.data import Dataset
from CT import get_ct_sample_size, get_ct, get_ct_raw_candidates
import random
from common_utils.logconfig import *
from seg_dset_utils import get_candidate_info_dict, get_candidate_info_list

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# base class (used for validation)
class SegmentationBase(Dataset):
    def __init__(self, dataset_dir_path:str,
                subsets_included:tuple = (0,1,2,3,4) ,val_stride = 0, val_set_bool = None, series_uid= None, context_slices = 3, full_ct=False):
        super(SegmentationBase, self).__init__()
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
        self.subsets_included = subsets_included
        self.context_volume = self.context_slices * 2 + 1
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
        

        self.sample_list = list() # all slices (positive & negative)
        for series_uid in self.subjects_list:
            num_indicies, positive_slices = get_ct_sample_size(series_uid, subsets_included)

            if self.full_ct: # get the whole volume slices
                self.sample_list += [(series_uid, slice_ind) 
                                     for slice_ind in range(num_indicies)]  
            else:
                self.sample_list += [(series_uid, slice_ind) 
                                     for slice_ind in positive_slices]
        
        # uploading individual candidates from each subject included in the subject list 
        self.candidate_list = get_candidate_info_list(dataset_dir_path, True, subsets_included)
        self.candidate_list = [sub for sub in self.candidate_list if sub.series_uid in self.subjects_list]

        self.pos_list = [sub for sub in self.candidate_list if sub.is_nodule] # positive candidates 

        log.info("{!r}: {} {} series, {} slices {} nodules".format(self, len(self.subjects_list), 
                                             {None: "general", True: "validation", False: "training"}[val_set_bool],
                                             len(self.sample_list),
                                             len(self.pos_list)))

    def __len__(self):
        return len(self.sample_list)
    
    def __getitem__(self, index):
        series_uid, slice_ind = self.sample_list[index % len(self.sample_list)] # circulation, even though it's useless with the __len__() definition above. 
        return self.slice_extract_with_context(series_uid, slice_ind)
    
    def slice_extract_with_context(self, series_uid, slice_ind):
        ct = get_ct(series_uid, self.subsets_included, usage = "segment")
        ct_context = torch.zeros((self.context_volume, 512, 512)) # taking self.context_slices above and below the target slice for context learning.

        start_ind = slice_ind - self.context_slices
        end_ind = slice_ind + self.context_slices + 1
        for i, context_idx in enumerate(range(start_ind, end_ind)):
            context_idx = max(context_idx, 0)
            context_idx = min(context_idx, ct.hu_arr.shape[0] - 1)
            ct_context[i] = torch.from_numpy(ct.hu_arr[context_idx])
        
        mask = torch.from_numpy(ct.positive_masks[slice_ind]).unsqueeze(0) # for number of channels
        return ct_context, mask, series_uid, slice_ind # last 2 returned values are used for logging purposes. 
    
"""
The reason for defining new separate dataset class for training is due to the difference in way of dealing with training and validation 
data, apart from just augmentation to compensate for dataset imbalance. In the training, we won't act upon the whole slice, just a patch 
around the candidate positve sample. This would help in solving the problem of data imbalance. Even though, this simple way of tackling imbalance 
issues would result in high false positive rates during validation.

"""
class TrainingSegmentDataset(SegmentationBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ratio_int = 1 # the alternation rate between the two classes 
        self.subsets_included = kwargs["subsets_included"]
        self.non_nodule_count = len(self.candidate_list) - len(self.pos_list)


    def __len__(self): 
        if self.ratio_int:
            candidates_count = len(self.candidate_list)  # at least  
            inbetwee_count = self.non_nodule_count  // self.ratio_int 
            if inbetwee_count >= len(self.pos_list):
                candidates_count += (inbetwee_count - len(self.pos_list))
            return candidates_count 
        else:
            return len(self.candidate_list) # without balancing

    def __getitem__(self, index):
        pos_candidate_info = self.pos_list[index % len(self.pos_list)] 
        return self.getitem_trainingcrop(pos_candidate_info)

    def getitem_trainingcrop(self, pos_candidate_info):
        extraction_size = (self.context_volume, 96, 96)
        pos_chunk, ct_chunk, icr_center = get_ct_raw_candidates(pos_candidate_info.series_uid, 
                                                                pos_candidate_info.center_xyz,
                                                                  extraction_size,
                                                                self.subsets_included,
                                                                  usage = "segment") 
        pos_mask = pos_chunk[3:4] # slicing maintain the third dimension. 
        # taking a 64 x 64 random crop of the chunk to add some kind of randomization (augmentation on the fly)
        row_offset = random.randrange(0, 32)
        col_offset = random.randrange(0, 32)
        ct_chop = torch.from_numpy(ct_chunk[:, row_offset:row_offset+64,
                                            col_offset:col_offset+64]).to(torch.float32)
        mask_chop = torch.from_numpy(pos_mask[:, row_offset:row_offset+64,
                                            col_offset:col_offset+64]).to(torch.long)
        
        slice_ind = icr_center.index

        return ct_chop, mask_chop, pos_candidate_info.series_uid, slice_ind  
    
        
    def shuffle_samples(self):
        random.shuffle(self.pos_list) # during training we will just use positive slices

