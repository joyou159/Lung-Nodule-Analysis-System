import sys 
import logging
from common_utils.logconfig import * 
import argparse
import numpy as np
import torch
from torch.utils.data import dataset, DataLoader
from CT import CT
import scipy.ndimage.morphology as morphology
import scipy.ndimage.measurements as measurements
from common_utils.util import * 
from seg_dset import SegmentationBase
from classifier_dset import LunaDataset 
from Unet import UNetWrapper
from NoduleClassifier import NoduleClassifier
import torch.nn as nn



log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class NoduleAnalysisApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            log.debug(sys.argv)
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()

        parser.add_argument('--batch-size',
            help='Batch size to use for validation (e.g. number of slices to be processed)',
            default=4,
            type=int,
        )
        parser.add_argument('--num-workers',
            help='Number of worker processes for background data loading',
            default=4,
            type=int,
        )

        parser.add_argument('series_uid',
            nargs='?',
            default=None,
            help="Series UID to use.",
        )

        parser.add_argument('--tb-prefix',
            default='nodule-analysis',
            help="Data prefix to use for Tensorboard run.",
        )

        self.args_list = parser.parse_args(sys_argv)
        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.seg_model_path = "models\seg_2024-10-11_20.40.34_segment.505150.best.state"
        self.nodule_model_path = "models\cls_2024-10-14_10.53.24_luna.best.state"
        self.malignancy_model_path = None

        self.seg_model, self.cls_model, self.malignancy_model = self.init_models()



    def init_models(self):
        seg_dict = torch.load(self.seg_model_path)

        seg_model = UNetWrapper(
                in_channels = 7,  
                num_classes = 1, # indicate the existence of nodule or not
                resolution_levels = 3,
                filters_power = 4,  # meaning the first layer will have (2**4) filters, each downsample layer will have double the start.
                padding = True, # to avoid losing information at the edges of the input. 
                batch_norm = True,  
                up_mode = "learnable", # upconv 
                dropout_rate = 0.2
        ) 

        seg_model.load_state_dict(seg_dict['model_state'])
        seg_model.eval()

        
        cls_dict = torch.load(self.nodule_model_path)

        model_cls = NoduleClassifier()
        cls_model = model_cls()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)

        if self.malignancy_model_path:
            model_cls = NoduleClassifier() # after being fine-tuned 
            malignancy_model = model_cls()
            malignancy_dict = torch.load(self.malignancy_model_path)
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None
        return seg_model, cls_model, malignancy_model



    def segment_ct(self, ct, series_uid):
        with torch.no_grad():
            output_a = np.zeros_like(ct.hu_arr, dtype=np.float32) 
            seg_dl = self.init_segmentation_dl(series_uid)  # this would allow us to interate over the ct slices in batches 
            
            for input_t, _, _, slice_ind_list in seg_dl:

                input_g = input_t.to(self.device) 
                prediction_g = self.seg_model(input_g) 

                for i, slice_ndx in enumerate(slice_ind_list):
                    output_a[slice_ndx] = prediction_g[i].cpu().numpy() 

            mask_a  = output_a > 0.5
            mask_a = morphology.binary_erosion(mask_a, iterations = 1) #  applies binary erosion as cleanup

        return mask_a
        
    def group_segmentation_output(self, ct, seg_output, series_uid):
        candidate_label, candidate_count = measurements.label(seg_output)
        IRC_center_list = measurements.center_of_mass(
            ct.hu_arr + 1001,  # this shifting to comply with the function expectations 
            labels = candidate_label, 
            index= np.arange(1, candidate_count+1) # stop is not included 
        ) 
        candidateInfo_list = list()
        for i, IRC_center in IRC_center_list:
            XYZ_center = irc2xyz(
            IRC_center, 
            ct.origin_xyz,
            ct.voxel_sizes,
            ct.transform_mat, 
            )
            
            candidateInfo_tup = CandidateInfoTuple(False, False, False, 0.0, series_uid, XYZ_center) 
            candidateInfo_list.append(candidateInfo_tup)

        return candidateInfo_list


    def classify_candidates(self, ct, candidateInfo_list):
        cls_dl = self.init_classification_dl(candidateInfo_list) 
        classification_list = list()
        for batch_ind, batch_tup in enumerate(cls_dl):
            input_t, _, _, series_list, center_list = batch_tup

            input_g = input_t.to(self.device)
            with torch.no_grad():
                _ , nodule_probabilities_g = self.cls_model(input_g)
                if self.malignancy_model is not None:
                    _, mal_probabilities_g = self.malignancy_model(input_g)
                else:
                    mal_probabilities_g = np.zeros_like(nodule_probabilities_g)
                
            zip_iter = zip(center_list, 
                           nodule_probabilities_g[:,1].tolist(),
                            mal_probabilities_g[:,1].tolist())
            for IRC_center, nodule_prob, mal_prob in zip_iter:
                XYZ_center = irc2xyz(
                        IRC_center, 
                        ct.origin_xyz,
                        ct.voxel_sizes,
                        ct.transform_mat, 
                        )
                cls_tup = (nodule_prob, mal_prob, XYZ_center, IRC_center)
                classification_list.append(cls_tup)
 
        return classification_list 
                
    def init_segmentation_dl(self, series_uid):
        # This would return the whole ct volume separated in slices
        seg_ds = SegmentationBase(
                DATASET_DIR_PATH, 
                subsets_included = (0,),
                context_slices=3,
                series_uid=series_uid,
                full_ct=True, 
            )
        
        seg_dl = DataLoader(
            seg_ds,
            batch_size=self.args_list.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.args_list.num_workers,
            pin_memory=self.use_cuda,
        )

        return seg_dl
    
    def init_classification_dl(self,  candidateInfo_list):
        cls_ds = LunaDataset(
                DATASET_DIR_PATH, 
                subsets_included = (0,),
                sortby_str='series_uid',
                candidateInfo_list=candidateInfo_list,
            )
        cls_dl = DataLoader(
            cls_ds,
            batch_size=self.args_list.batch_size * (torch.cuda.device_count() if self.use_cuda else 1),
            num_workers=self.args_list.num_workers,
            pin_memory=self.use_cuda,
        )

        return cls_dl
    
