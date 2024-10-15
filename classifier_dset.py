import torch 
from torch.utils.data import Dataset 
import copy 
import random
from classifier_dset_utils import *

class LunaDataset(Dataset):
    def __init__(self, dataset_dir_path:str,
                subsets_included:tuple = (0,1,2,3,4),
                 val_stride=0,
                 val_set_bool=None,
                 ratio_int=0,
                 series_uid=None,
                 sortby_str='random',
                 augmentation_dict=None,
                 candidateInfo_list=None,
            ):
        
        """
            Initialize training or validation dataset over the entire subjects of specific subject 
            by skipping over (CT scans not candidates) using a specified validation stride. 
            
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
                - augmentation_dict: specifying the kind of transformations included in the augmentation process.
                - candidateInfo_list: list all the information associated to each candidate nodule in the training set.            
        """
        self.ratio_int = 1 # default
        self.augmentation_dict = augmentation_dict
        if self.augmentation_dict is not None:
            self.augmentation_model = AugmentationCandidate(augmentation_dict) 
        

        if candidateInfo_list: 
            self.candidateInfo_list = copy.copy(candidateInfo_list)
            self.use_cache = False # don't use cache in this case because you would be working on just one ct volume. 
            # thus, it would be left suspended in the memory during the accessing its entries. 
        else:
            self.candidateInfo_list = copy.copy(get_candidate_info_list(DATASET_DIR_PATH, required_on_desk=True,subsets_included = subsets_included))
            self.use_cache = True

        if series_uid:
            self.series_list = [series_uid]
        else:
            self.series_list = sorted(set(candidateInfo_tup.series_uid for candidateInfo_tup in self.candidateInfo_list))

        if val_set_bool:
            assert val_stride > 0, val_stride
            self.series_list = self.series_list[::val_stride]
            assert self.series_list
        elif val_stride > 0:
            del self.series_list[::val_stride]
            assert self.series_list

        series_set = set(self.series_list)
        self.candidateInfo_list = [x for x in self.candidateInfo_list if x.series_uid in series_set]

        if sortby_str == 'random':
            random.shuffle(self.candidateInfo_list)
        elif sortby_str == 'series_uid':
            self.candidateInfo_list.sort(key=lambda x: (x.series_uid, x.center_xyz))
        elif sortby_str == 'label_and_size':
            pass
        else:
            raise Exception("Unknown sort: " + repr(sortby_str))

        self.neg_list = \
            [nt for nt in self.candidateInfo_list if not nt.is_nodule]
        self.pos_list = \
            [nt for nt in self.candidateInfo_list if nt.is_nodule]
        self.ben_list = \
            [nt for nt in self.pos_list if not nt.is_malignant]
        self.mal_list = \
            [nt for nt in self.pos_list if nt.is_malignant]

        log.info("{!r}: {} {} samples, {} neg, {} pos, {} ratio".format(
            self,
            len(self.candidateInfo_list),
            "validation" if val_set_bool else "training",
            len(self.neg_list),
            len(self.pos_list),
            '{}:1'.format(self.ratio_int) if self.ratio_int else 'unbalanced'
        ))

    def shuffle_samples(self):
        if self.ratio_int:
            random.shuffle(self.candidateInfo_list)
            random.shuffle(self.neg_list)
            random.shuffle(self.pos_list)
            random.shuffle(self.ben_list)
            random.shuffle(self.mal_list)

    def __len__(self):
        if self.ratio_int:
            candidates_count = len(self.candidateInfo_list)  # at least  
            inbetwee_count = len(self.neg_list) // self.ratio_int 
            if inbetwee_count >= len(self.pos_list):
                candidates_count += (inbetwee_count - len(self.pos_list))
            return candidates_count 
        else:
            return len(self.candidateInfo_list) 

    def __getitem__(self, ndx):
        if self.ratio_int:
            pos_ndx = ndx // (self.ratio_int + 1)

            if ndx % (self.ratio_int + 1):
                neg_ndx = ndx - 1 - pos_ndx
                neg_ndx %= len(self.neg_list)
                candidateInfo_tup = self.neg_list[neg_ndx]
            else:
                pos_ndx %= len(self.pos_list)
                candidateInfo_tup = self.pos_list[pos_ndx]
        else:
            candidateInfo_tup = self.candidateInfo_list[ndx]

        return self.sample_from_candidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.is_nodule
        )

    def sample_from_candidateInfo_tup(self, candidateInfo_tup, label_bool):
        width_irc = (32, 48, 48)

        if self.augmentation_dict:
            candidate_t, center_irc = self.augmentation_model(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
                self.use_cache,
            )
        elif self.use_cache:
            candidate_a, center_irc = get_ct_raw_candidates(
                candidateInfo_tup.series_uid,
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)
        else:
            ct = get_ct(candidateInfo_tup.series_uid)
            candidate_a, center_irc = ct.get_raw_candidate_nodule(
                candidateInfo_tup.center_xyz,
                width_irc,
            )
            candidate_t = torch.from_numpy(candidate_a).to(torch.float32)
            candidate_t = candidate_t.unsqueeze(0)

        label_t = torch.tensor([False, False], dtype=torch.long)

        if not label_bool:
            label_t[0] = True
            index_t = 0
        else:
            label_t[1] = True
            index_t = 1

        return candidate_t, label_t, index_t, candidateInfo_tup.series_uid, IRC_tuple(*center_irc)



class MalignantLunaDataset(LunaDataset):
    def __len__(self):
        if self.ratio_int:
            return 100000
        else:
            return len(self.ben_list + self.mal_list)

    def __getitem__(self, ndx):
        if self.ratio_int:
            if ndx % 2 != 0:
                candidateInfo_tup = self.mal_list[(ndx // 2) % len(self.mal_list)]
            elif ndx % 4 == 0:
                candidateInfo_tup = self.ben_list[(ndx // 4) % len(self.ben_list)]
            else:
                candidateInfo_tup = self.neg_list[(ndx // 4) % len(self.neg_list)]
        else:
            if ndx >= len(self.ben_list):
                candidateInfo_tup = self.mal_list[ndx - len(self.ben_list)]
            else:
                candidateInfo_tup = self.ben_list[ndx]

        return self.sample_from_candidateInfo_tup(
            candidateInfo_tup, candidateInfo_tup.is_malignant
        )