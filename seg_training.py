import argparse
import sys
import os 
import datetime
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from util import *
from logconfig import *
from segment_dset import *
from Unet import UNetWrapper 
from segment_dset_utils import SegmentationAugmentation
from torch.utils.tensorboard import SummaryWriter

# for Tensorboard logging 
METRICS_LOSS_NDX = 0
METRICS_TP_NDX = 1
METRICS_FN_NDX = 2
METRICS_FP_NDX = 3

METRICS_SIZE = 4 

class SegmentationTrainingApp():
    def __init__(self, sys_argv = None):
        if sys_argv is None: # if the caller doesn't provide parameters, we get them from the command line.
             sys_argv = sys.argv[1:]
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_workers", 
                            help="Number of worker processes for background data loading (number of cores)", 
                            default = 8,
                            type=int)

        parser.add_argument('--batch-size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )

        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        
        parser.add_argument('--subsets-included',
            help='The number of subsets included in the training process',
            default=(0,1,2,3,4),
            type=tuple,
        )

        parser.add_argument('--tb-prefix',
            default='tensorboard-prefix',
            help="Data prefix to use for Tensorboard run.",
        )
        # -------------------------- data augmentation arguments -----------------------------------
        parser.add_argument('--augmented',
            help="Augment the training data.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-flip',
            help="Augment the training data by randomly flipping the data left-right, up-down, and front-back.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-offset',
            help="Augment the training data by randomly offsetting the data slightly along the X and Y axes.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-scale',
            help="Augment the training data by randomly increasing or decreasing the size of the candidate.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-rotate',
            help="Augment the training data by randomly rotating the data around the head-foot axis.",
            action='store_true',
            default=False,
        )
        parser.add_argument('--augment-noise',
            help="Augment the training data by randomly adding noise to the data.",
            action='store_true',
            default=False,
        )
        #-------------------------------------------------------------------------
        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='segment',
        )
        # --------------------- dataset balancing --------------------------------
        parser.add_argument('--balanced',
            help="Balance the training data to half positive, half negative.",
            action='store_true',
            default=False,
            )

        self.args_list  = parser.parse_args(sys_argv)

        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S') # to identify running times

        # tensorboard settings
        self.train_writer = None
        self.val_writer = None
        # this would be the x-axis of the metrics plots (more descriptive)
        self.totalTrainingSamples_count = 0


        self.augmentation_dict = {}
        if self.args_list.augmented:
            if self.args_list.augment_flip:
                self.augmentation_dict['flip'] = True
            if self.args_list.augment_offset:
                self.augmentation_dict['offset'] = 0.1
            if self.args_list.augment_scale:
                self.augmentation_dict['scale'] = 0.2
            if self.args_list.augment_rotate:
                self.augmentation_dict['rotate'] = True
            if self.args_list.augment_noise: # this value must be chosen carefully, because it might result in a disasters 
                self.augmentation_dict['noise'] = 25.0 # max density deviation is 20 (adjusted range used [-1000, 1000])


        self.use_cuda =  torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # generic 
        
        self.segmentation_model, self.augmentation_model = self.init_model() 
        self.optimizer = self.init_optimizer()


    def init_model(self):
        segmentation_model = UNetWrapper(
                in_channel = 7,  
                num_classes = 1, # indicate the existence of nodule or not
                resolution_levels = 3,
                filters_power = 4,  # meaning the first layer will have (2**4) filters, each downsample layer will have double the start.
                padding = True, # to avoid losing information at the edges of the input. 
                batch_norm = True,  
                up_mode = "learnable" # upconv 
        ) 

        augmentation_model = SegmentationAugmentation(**self.augmentation_dict) 
        # set up GPU acceleration.besides, multi-GPU training using DataParallel if possible.
        if self.use_cuda:
            log.info(f"Using CUDA; {torch.cuda.device_count()} devices.") 
            if torch.cuda.device_count() > 1:
                segmentation_model = nn.DataParallel(segmentation_model)
                augmentation_model = nn.DataParallel(augmentation_model)
            segmentation_model = segmentation_model.to(self.device)
            augmentation_model = augmentation_model.to(self.device)

        return segmentation_model, augmentation_model
    

    def init_optimizer(self): 
        return torch.optim.Adam(self.segmentation_model.parameters()) 
    

    def init_data_loader(self, val_set_bool = False):
        if val_set_bool: # no balancing for validation (real-world isn't balanced anyway)
            dataset = SegmentationBase(DATASET_DIR_PATH, 
                                  self.args_list.subsets_included, 
                                  val_set_bool=True,
                                  val_stride=10,
                                  context_slices = 3, 
                                  full_ct=False)
        else: # ratio_int = 1 (alternating)
            dataset = TrainingSegmentDataset(DATASET_DIR_PATH,
                                 self.args_list.subsets_included,
                                 val_set_bool=False,
                                 val_stride=10,
                                 ratio_int=int(self.args_list.balanced))
            
        batch_size = self.args_list.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # each GPU has its own batch 
        
        data_loader = DataLoader(dataset, batch_size=batch_size,
                               num_workers=self.args_list.num_workers  # this refers to the num of CPU processes to load data to memory in parallel 
                               ,pin_memory=self.use_cuda) # transfer the data in the memory to the GPU quickly

        return data_loader # train or validation (accordingly)

    
    def initTensorboardWriters(self):
        if self.train_writer is None: 
            log_dir = os.path.join('runs', self.args_list.tb_prefix, self.time_str) 

            self.train_writer = SummaryWriter(
                log_dir=log_dir + '-train_seg-' + self.args_list.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_seg-' + self.args_list.comment)
            

    def training_epoch(self, epoch_ndx, train_dl):
        self.segmentation_model.train() 
        train_dl.dataset.shuffle_samples() # shuffle the training data per epoch 

        # initialize empty metrics array per sample to keep track the performance per sample. This would give us a nice insights into 
        # when our model fails.
        train_metrics_per_sample = torch.zeros(
            METRICS_SIZE, 
            len(train_dl.dataset), 
            device=self.device
        )
        
        # epoch iteration with loggings 
        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers)
        
        for batch_ndx, curr_batch in batch_iter:
            self.optimizer.zero_grad() # remove leftover gradient tensors
            # custom loss function to handle the separation between training samples per batch. 
            loss = self.compute_batch_loss(batch_ndx, curr_batch, train_dl.batch_size, train_metrics_per_sample)

            loss.backward() 
            self.optimizer.step() 

        self.totalTrainingSamples_count += len(train_dl.dataset) # using this as the x-axis at each epoch 
        # instead of using epoch ndx as the x-axis value, we tend to use the number of batches as a more representative value 
        # for the sake of comparison with less or more size batchs runs.

        return train_metrics_per_sample.to("cpu") # release space from the gpu 


    def validation_epoch(self, epoch_ndx, val_dl):
        with torch.no_grad(): 
            self.segmentation_model.eval()
            val_metrics_per_sample = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device = self.device
            )
             
            batch_iter = enumerateWithEstimate(
            val_dl,
            f"E{epoch_ndx} Validation",
            start_ndx=val_dl.num_workers)

            for batch_ndx, curr_batch in batch_iter:
                self.compute_batch_loss(batch_ndx, curr_batch, val_dl.batch_size, val_metrics_per_sample) 
                # no need to store the loss as there's no update required 
            
        return val_metrics_per_sample.to("cpu")


    def dice_loss(self, predictions, labels, epsilon = 1):
        dice_predictions = predictions.sum(dim=[1, 2, 3]) # entry-wise diceloss computation  
        dice_labels =  labels.sum(dim=[1,2,3])   
        dice_overlap = (predictions * labels).sum(dim=[1,2,3]) 
        # epsilon to account for when we accidentally have neither predictions nor labels (in the case of full_ct mode during training)
        dice_coff = (2 * dice_overlap + epsilon) / (dice_predictions + dice_labels + epsilon) 

        return 1 - dice_coff
    
    def compute_batch_loss(self, batch_ndx, curr_batch, batch_size, metrics_per_sample, fn_weighting_factor = 8 ,classify_threshold = 0.5):
        # training data loader returns (ct_context, mask, series_uid, slice_ind)
        # validation data loader returns (ct_chop, mask_chop, pos_candidate_info.series_uid, slice_ind) 
        input_t, masks_t, series_uid, slice_ind = curr_batch

        # Transfer to GPU ('non_blcoking' means transfer the batch data in asynch (speed improvement))
        input_g = input_t.to(self.device, non_blocking = True) 
        masks_g = masks_t.to(self.device, non_blocking = True)

        if self.segmentation_model.training:
            input_g, masks_g = self.augmentation_model(input_g, masks_g)

        predictions_g = self.segmentation_model(input_g)

        dice_loss_g = self.dice_loss(predictions_g, masks_g)
        fnloss_g = self.dice_loss(predictions_g * masks_g, masks_g) # this loss remove false positive from loss computation (just false negative)  
        
        # by this way, you are 9 times emphasizing on false negative than false postive. Makes sense in the context of our problem.
        loss_function = dice_loss_g + fnloss_g * fn_weighting_factor # you can choose other weighting factor  

        start_ind = batch_size * batch_ndx
        end_ind = start_ind + input_g.size(0) 

        with torch.no_grad():
            predictions_bool = (predictions_g[:,0:1] > classify_threshold).to(torch.float32) # the slicing here for maintaining 2nd dim 

            tp = (predictions_bool * masks_g).sum(dim=[1,2,3])
            fn = ((1 - predictions_bool) * masks_g).sum(dim=[1,2,3])
            fp = ((~masks_g) * predictions_bool).sum(dim=[1,2,3])

            metrics_per_sample[METRICS_LOSS_NDX, start_ind:end_ind] = dice_loss_g 
            metrics_per_sample[METRICS_TP_NDX, start_ind:end_ind] = tp 
            metrics_per_sample[METRICS_FN_NDX, start_ind:end_ind] =  fn
            metrics_per_sample[METRICS_FP_NDX, start_ind:end_ind] = fp

        return loss_function









