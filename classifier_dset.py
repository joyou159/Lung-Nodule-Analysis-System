
import torch 
from torch.utils.data import Dataset 
import copy 
import random
from classifier_dset_utils import *

class LunaDataset(Dataset):
    def __init__(self, dataset_dir_path = DATASET_DIR_PATH,
                subsets_included:tuple = (0,1,2,3,4) ,
                val_stride:int = 0,
                val_set_bool:bool = None, 
                ratio_int:int = 0 ,
                series_uid:str = None,
                sortby_str:str='random',
                augmentation_dict = None):
        """
            Initialize training or validation dataset over the entire candidate nodules by skipping over
            using a specified validation stride.
            
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
        self.candidates_info_list = copy.copy(get_unified_candidate_info_list(dataset_dir_path , required_on_desk=True,subsets_included = subsets_included)) # to isolate the cached list 
        
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
            candidate_arr, irc_center = get_ct_raw_candidates(candidate_info.series_uid ,candidate_info.center_xyz, irc_width, "classifier")
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
        

