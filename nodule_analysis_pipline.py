import sys 
import logging
from common_utils.logconfig import * 
import argparse
import numpy as np
import torch
from torch.utils.data import dataset, DataLoader
from CT import CT, get_ct
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

def match_and_score(detections, truth, threshold=0.5, matching_threshold=0.7):
    """
    Computes a 3x4 confusion matrix to evaluate the performance of a detection algorithm
    for identifying nodules in medical imaging. The matrix contains counts for various
    categories based on the type of ground truth and detection outcomes.

    Args:
        detections (list): A list of detected nodules, each with these info (nodule_prob, mal_prob, XYZ_center, IRC_center).

        truth (list): A list of ground truth annotations for nodules, each with these info 
        (is_nodule, has_annotations, is_malignant, diameter_mm, series_uid, center_xyz)

        threshold (float, optional): classification threshold for nodule and malignancy classifiers.

    Returns:
        numpy.ndarray: A 3x4 confusion matrix where:
            - Rows represent ground truth categories: [Non-Nodules, Benign, Malignant].
            - Columns represent detection outcomes:
                [Not Detected, Detected by Segmentation, Detected as Benign, Detected as Malignant].

    Notes:
        - If multiple detections match a single ground truth nodule, the detection with
          the "highest" classification (based on severity) is considered.
        - If a single detection matches multiple ground truth annotations, it counts for
          all of them.
    """
    true_nodules = [c for c in truth if c.is_nodule]
    truth_diams = np.array([c.diameter_mm for c in true_nodules])
    truth_xyz = np.array([c.center_xyz for c in true_nodules]) 

    detected_xyz = np.array([n[2] for n in detections]) 
    # detection classes will contain
    # 1 -> detected by seg but filtered by cls
    # 2 -> detected as benign nodule (or nodule if no malignancy model is used)
    # 3 -> detected as malignant nodule (if applicable) 

    detected_classes = np.array([1 if d[0] < threshold # nodule classificaiotn check 
                                 else (2 if d[1] < threshold # malignancy classificatiion check 
                                       else 3) for d in detections]) 
    

    confusion = np.zeros((3, 4), dtype=np.int32)

    if len(detected_xyz) == 0: 
        for tn in true_nodules:
            confusion[2 if tn.is_malignant else 1, 0] += 1  # increment for benign and malignant misses 
    elif len(truth_xyz) == 0: 
        for dc in detected_classes:
            confusion[0, dc] += 1 
    else:
        # truth_xyz[:, None] -> of shape (num_truth, 1, 3) & detected_xyz[None] -> of shape (1, num_detected, 3) broadcasting is valid 
        # note that the distance bewteen the detection and ground truth is normalized by the ground truth diameter to make it a function of size.
        normalized_dists = np.linalg.norm(truth_xyz[:, None] - detected_xyz[None], ord=2, axis=-1) / truth_diams[:, None]  #  returned shape (num_truth, num_detected)
        matches = (normalized_dists < matching_threshold) 
        
        # mark all the detection as matched until otherwise is figured out  
        unmatched_detections = np.ones(len(detections)).astype(bool)

        # mark the lable that our system gives to each true nodule 
        # 0-> non-detected  
        # 1-> detected (discarded later in nodule classifier)
        # 2-> detected and marked as benign in the malignancy classifier. 
        # 3-> detected and marked as malignant in the malignancy classifier.
        # all set to non-detected yet
        matched_true_nodules = np.zeros(len(true_nodules), dtype=np.int32) 

        for i_tn, i_detection in zip(*matches.nonzero()): # (num_truth, num_detected) iterations on all the matched (non-zero) detections 
            # given the current true nodule, what is the detection that reaches the most far and near enough (govern by the match threshold)
            matched_true_nodules[i_tn] = max(matched_true_nodules[i_tn], detected_classes[i_detection]) 
            unmatched_detections[i_detection] = False 

        for ud, dc in zip(unmatched_detections, detected_classes):
            if ud: # any unmached detection
                confusion[0, dc] += 1 # increment in (non-nodule, dc) cell, where dc-> (0, 1, 2, 3) 
        for tn, dc in zip(true_nodules, matched_true_nodules):
            confusion[2 if tn.is_malignant else 1, dc] += 1 

    return confusion




def print_confusion(label, confusions, do_mal):

    """
    - To summarize our system output   (if there is a malignancy check)

                    | Complete Miss  | Filtered Out   | Pred. Benign   | Pred. Malignant
    ---------------------------------------------------------------------------------------
    Non-Nodules       |                | value          | value          | value
    Benign            | value          | value          | value          | value
    Malignant         | value          | value          | value          | value

    """

    row_labels = ['Non-Nodules', 'Benign', 'Malignant']

    if do_mal:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Benign', 'Pred. Malignant']
    else:
        col_labels = ['', 'Complete Miss', 'Filtered Out', 'Pred. Nodule']
        confusions[:, -2] += confusions[:, -1] # accumulate the last two columns to form the 'Pred. Nodule' column 
        confusions = confusions[:, :-1] 
    cell_width = 16
    f = '{:>' + str(cell_width) + '}'
    print(label)
    print(' | '.join([f.format(s) for s in col_labels]))
    for i, (l, r) in enumerate(zip(row_labels, confusions)):
        r = [l] + list(r)
        if i == 0:
            r[1] = ''
        print(' | '.join([f.format(i) for i in r]))


 
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

        parser.add_argument('--run-validation',
            help='Run over validation rather than a single CT.',
            action='store_true',
            default=False,
        )

        parser.add_argument('--include-train',
            help="Include data that was in the training set. (default: validation data only)",
            action='store_true',
            default=False,
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

        self.seg_model_path = "/kaggle/input/nodule_detection_segmentation_model/pytorch/default/1/seg_2024-10-11_20.40.34_segment.505150.best.state"
        self.nodule_model_path = "/kaggle/input/nodule_classifier_model/pytorch/default/1/cls_2024-10-14_10.53.24_luna.best.state"
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

        cls_model = NoduleClassifier()
        cls_model.load_state_dict(cls_dict['model_state'])
        cls_model.eval()

        if self.use_cuda:
            if torch.cuda.device_count() > 1:
                seg_model = nn.DataParallel(seg_model)
                cls_model = nn.DataParallel(cls_model)

            seg_model.to(self.device)
            cls_model.to(self.device)

        if self.malignancy_model_path:
            malignancy_model =  NoduleClassifier() # after being fine-tuned 
            malignancy_dict = torch.load(self.malignancy_model_path)
            malignancy_model.load_state_dict(malignancy_dict['model_state'])
            malignancy_model.eval()
            if self.use_cuda:
                malignancy_model.to(self.device)
        else:
            malignancy_model = None
        return seg_model, cls_model, malignancy_model


    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.args_list}") 

        val_ds = LunaDataset(DATASET_DIR_PATH, 
                subsets_included = (0,),val_stride=10, val_set_bool=True) 
        
        # get the series for all the validation candidates
        val_set = set(candidateInfo_tup.series_uid for candidateInfo_tup in val_ds.candidateInfo_list) 

    
        candidateInfo_list = get_candidate_info_list(DATASET_DIR_PATH, required_on_desk=True,subsets_included = (0,))

        # get the whole positive nodules, whether in the validaiton or training set 
        positive_set = set(candidateInfo_tup.series_uid for candidateInfo_tup in candidateInfo_list if candidateInfo_tup.is_nodule) 

        # if there is specific series_uid provided
        if self.args_list.series_uid:
            series_set = set(self.args_list.series_uid.split(",")) 
        else:
            series_set = set(
                candidateInfo_tup.series_uid
                for candidateInfo_tup in candidateInfo_list
            )

        if self.args_list.include_train:
            train_list = sorted(series_set - val_set)

        else:
            train_list = list() 
        
        val_list = sorted(series_set & val_set) 
        
        candidateInfo_dict = get_candidate_info_dict(DATASET_DIR_PATH, required_on_desk=True,subsets_included = (0,))
        series_iter = enumerateWithEstimate(
            val_list + train_list,
            "Series",
        )  

        all_confusion = np.zeros((3, 4), dtype=np.int32) 

        for _, series_uid in series_iter:
            ct = get_ct(series_uid, subset_included = (0,), usage = "segment") 
            # pipline start 
            mask_a = self.segment_ct(ct, series_uid) 
            groupingInfo_list = self.group_segmentation_output(ct, mask_a, series_uid) 
            classifications_list = self.classify_candidates(ct, groupingInfo_list) 
            # pipline end  

            if not self.args_list.run_validation: 
                print(f"found nodule candidates in {series_uid}:")
                for prob, prob_mal, center_xyz, center_irc in classifications_list:
                    if prob > 0.5:
                        s = f"nodule prob {prob:.3f}, "
                        if self.malignancy_model:
                            s += f"malignancy prob {prob_mal:.3f}, "
                        s += f"center xyz {center_xyz}"
                        print(s)

            if series_uid in candidateInfo_dict:
                one_confusion = match_and_score(
                    classifications_list, candidateInfo_dict[series_uid]
                ) 

                all_confusion += one_confusion 

                print_confusion(series_uid, one_confusion, self.malignancy_model is not None)

        print_confusion("Total", all_confusion, self.malignancy_model is not None)  



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
        for i, IRC_center in enumerate(IRC_center_list):
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
                    mal_probabilities_g = np.zeros_like(nodule_probabilities_g.cpu().numpy())
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
    
    def init_classification_dl(self, candidateInfo_list):
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
    