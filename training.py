import argparse
import sys
import datetime
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from util import *
from logconfig import *
from dsets import *
from model import LunaModel 
from torch.utils.tensorboard import SummaryWriter


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

# macros for keeping track individual samples records in each batch.
METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_LOSS_NDX=2
METRICS_SIZE = 3

# fully fledged command-line application, meaning  It will
# parse command-line arguments, have a full-featured --help command, and be easy to
# run in a wide variety of environments.

class LunaTrainingApp():
    def __init__(self, sys_argv = None):
        if sys_argv is None: # if the caller doesn't provide parameters, we get them from the command line.
             sys_argv = sys.argv[1:]
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--num_workers", 
                            help="Number of worker processes for background data loading (number of cores)", 
                            default = 8,
                            type=int)

        parser.add_argument('--batch_size',
            help='Batch size to use for training',
            default=32,
            type=int,
        )

        parser.add_argument('--epochs',
            help='Number of epochs to train for',
            default=1,
            type=int,
        )
        
        parser.add_argument('--model_path',
            help='The path to the existing model',
            default=None,
            type=str,
        )
        
        parser.add_argument('--subsets_included',
            help='The number of subsets included in the training process',
            default=(0,1,2,3,4),
            type=tuple,
        )

        parser.add_argument('--tb-prefix',
            default='tensorboard-prefix',
            help="Data prefix to use for Tensorboard run.",
        )

        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='luna',
        )

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
        self.totalTrainingSamples_count = 0


        self.use_cuda =  torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        # generic 
        
        self.model = self.init_model(self.args_list.model_path) 
        self.optimizer = self.init_optimizer()


    def init_model(self, model_path = None):
        model = LunaModel()
        if model_path is not None:
            model.load_state_dict(torch.load(model_path))
        if self.use_cuda:
            log.info(f"Using CUDA, with {torch.cuda.device_count()} devices")
            if torch.cuda.device_count() > 1:  # if the system on which the training is conducted has more than just one GPU
                # then we would distribute the workload on them then collect and
                # resync parameter updates and so on.      
                model = nn.DataParallel(model)
            model = model.to(self.device) 
        return model 
    

    def init_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.001) # tunable 


    def init_data_loader(self, val_set_bool = False):
        if val_set_bool: # no balancing for validation (real-world isn't balanced anyway)
            dataset = LunaDataset(DATASET_DIR_PATH, self.args_list.subsets_included, val_set_bool=val_set_bool, val_stride=10)
        else: # ratio_int = 1 (alternating)
            dataset = LunaDataset(DATASET_DIR_PATH, self.args_list.subsets_included, val_set_bool=val_set_bool, val_stride=10, ratio_int=int(self.args_list.balanced))
        batch_size = self.args_list.batch_size
        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # each GPU has its own batch 
        
        data_loader = DataLoader(dataset, batch_size=batch_size,
                               num_workers=self.args_list.num_workers  # this refers to the num of CPU processes to load data to memory in parallel 
                               ,pin_memory=self.use_cuda) # transfer the data in the memory to the GPU quickly

        return data_loader


    def initTensorboardWriters(self):
        if self.train_writer is None: 
            log_dir = os.path.join('runs', self.args_list.tb_prefix, self.time_str) 

            self.train_writer = SummaryWriter(
                log_dir=log_dir + '-train_cls-' + self.args_list.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.args_list.comment)


    def main(self):
        log.info(f"Starting {type(self).__name__}, {self.args_list}")

        train_dl = self.init_data_loader()
        val_dl = self.init_data_loader(val_set_bool = True)

        for epoch_ndx in range(1, self.args_list.epochs + 1): 

            log.info(f"epoch no.{epoch_ndx} of {self.args_list.epochs} -- (train_dl/val_dl): {len(train_dl)}/{len(val_dl)} -- with batch size of {self.args_list.batch_size} on {torch.cuda.device_count()} GPUs")

            # training  
            train_metrics_per_sample = self.training_epoch(epoch_ndx, train_dl)  
            self.log_metrics(epoch_ndx, "train", train_metrics_per_sample) 
            # validation 
            val_metrics_per_sample = self.validation_epoch(epoch_ndx, val_dl)
            self.log_metrics(epoch_ndx, "val", val_metrics_per_sample)



        if hasattr(self, 'train_writer'):
            self.train_writer.close()
            self.val_writer.close()


    def training_epoch(self,epoch_ndx , train_dl):
        self.model.train() # set the model on the training mode 
        train_dl.dataset.shuffle_samples() # shuffle the training data per epoch 
        # initialize empty metrics array per sample to keep track the performance per sample. This would give us a nice insights into 
        # when our model fails.
        train_metrics_per_sample = torch.zeros(
            METRICS_SIZE, 
            len(train_dl.dataset), 
            device=self.device
        )
        
        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers)
        
        for batch_ndx, curr_batch in batch_iter:
            self.optimizer.zero_grad() # remove leftover gradient tensors
            # custom loss function to handle the separation between training samples per batch. 
            loss_val = self.compute_batch_loss(batch_ndx, curr_batch, train_dl.batch_size, train_metrics_per_sample)

            loss_val.backward() # compute gradients. 
            self.optimizer.step() # update parameters.

        # tensorboard settings 
        self.totalTrainingSamples_count += len(train_dl.dataset) # using this as the x-axis at each epoch 
        # instead of using epoch ndx as the x-axis value, we tend to use the number of batches as a more representative value 
        # for the sake of comparison with less or more size batchs runs.

        return train_metrics_per_sample.to("cpu") # release space from the gpu 


    def validation_epoch(self, epoch_ndx, val_dl):
        with torch.no_grad(): 
            self.model.eval()
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


    def compute_batch_loss(self, batch_ndx, curr_batch, batch_size, train_metrics_per_sample):
        """ Computes the loss over a batch of samples. By recording the label, prediction, and loss for each and every training 
            as well as validation sample, we have a wealth of detailed information we can use to investigate
            the behavior of our model.
        """
        samples, labels, series_uids, irc_centers = curr_batch 

        # upload the data on the GPUs 
        samples = samples.to(self.device, non_blocking = True) # non_blcoking means transfer the batch data in asynch (speed improvement)
        labels = labels.to(self.device, non_blocking = True) 

        logits, probs = self.model(samples)  
        loss_fun = nn.CrossEntropyLoss(reduction="none") # reduction = "none" means that return the loss per sample, not per batch

        loss_val = loss_fun(
            logits, labels[:,1] # since labels are one-hot encoded, and it's a binary classification problem  
        )
        # since train_metrics_per_sample is a tensor for the whole epoch, meaning all the batches samples contribution and effect 
        # will be recorded, so we need effective slicing. 
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + labels.size(0) 

        train_metrics_per_sample[METRICS_LABEL_NDX, start_ndx:end_ndx] = labels[:,1].detach() # detach from the computaitonal graph 
        # phrased differently, don't keep gradient. 
        train_metrics_per_sample[METRICS_PRED_NDX, start_ndx:end_ndx] = probs[:,1].detach()
        train_metrics_per_sample[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_val.detach()

        return loss_val.mean() # recombine the loss of each individual sample to get the overall loss of the batch


    def log_metrics(self, epoch_ndx:int, mode:str, metrics_per_sample, classification_thr = 0.5):

        self.initTensorboardWriters() 

        log.info(f"E{epoch_ndx}, {type(self).__name__}")
        
        # non-nodule class
        neg_label_mask = metrics_per_sample[METRICS_LABEL_NDX] <= classification_thr   
        neg_pred_mask = metrics_per_sample[METRICS_PRED_NDX] <= classification_thr  
        # nodule class 
        pos_label_mask = ~neg_label_mask
        pos_pred_mask = ~neg_pred_mask

        neg_count = int(neg_label_mask.sum()) 
        pos_count = int(pos_label_mask.sum())

        # ture negative and true postive respectively 
        neg_correct = int((neg_label_mask & neg_pred_mask).sum())
        pos_correct = int((pos_label_mask & pos_pred_mask).sum())

        # false negative and false postive 
        false_neg = pos_count - pos_correct  
        false_pos = neg_count - neg_correct 

        metrics_dict = dict()

        # losses  
        metrics_dict["loss/all"] = metrics_per_sample[METRICS_LOSS_NDX].mean() 
        metrics_dict["loss/neg"] = metrics_per_sample[METRICS_LOSS_NDX, neg_label_mask].mean()
        metrics_dict["loss/pos"] = metrics_per_sample[METRICS_LOSS_NDX, pos_label_mask].mean()
        # accuracy 
        metrics_dict["acc/all"] = ((neg_correct + pos_correct) / np.float32(metrics_per_sample.shape[1])) * 100  
        metrics_dict["acc/neg"] = (neg_correct / np.float32(neg_count)) * 100  
        metrics_dict["acc/pos"] = (pos_correct / np.float32(pos_count)) * 100  
        # precision & recall 
        metrics_dict["pr/precision"] = (pos_count / np.float32(pos_count + false_pos)) # don't flag as nodule unless you are sure 
        metrics_dict["pr/recall"] = (pos_count / np.float32(pos_count + false_neg)) # don't miss out any nodules under any circumstances
        # F1 score (condensing both precision and recall in one metric)
        metrics_dict["pr/f1_score"] = \
        2 * (metrics_dict["pr/precision"] * metrics_dict["pr/recall"]) / (metrics_dict["pr/precision"] + metrics_dict["pr/recall"])


        log.info(f'E{epoch_ndx} -- {mode} -- {metrics_dict["loss/all"]:.4f} overall loss -- {metrics_dict["acc/all"]:.1f}% accuracy -- {metrics_dict["pr/precision"]:.4f} precision -- {metrics_dict["pr/recall"]:.4f} recall -- {metrics_dict["pr/f1_score"]:.4f} f1 score')
        log.info(f'E{epoch_ndx} {mode}  {metrics_dict["loss/neg"]:.4f} negative loss {metrics_dict["acc/neg"]:.1f}% accuracy, meaning {neg_correct} of {neg_count} are correct')
        log.info(f'E{epoch_ndx} {mode}  {metrics_dict["loss/pos"]:.4f} positive loss {metrics_dict["acc/pos"]:.1f}% accuracy, meaning {pos_correct} of {pos_count} are correct')
        


        # tensorboard reporting 
        writer = getattr(self, mode + '_writer') # SummaryWriter object 

        for key, value in metrics_dict.items():
            writer.add_scalar(key, value, self.totalTrainingSamples_count) 

        writer.add_pr_curve(
            'pr',
            metrics_per_sample[METRICS_LABEL_NDX],
            metrics_per_sample[METRICS_PRED_NDX],
            self.totalTrainingSamples_count,
        ) 

        bins = [x/50.0 for x in range(51)]

        neg_hist_mask = neg_label_mask & (metrics_per_sample[METRICS_PRED_NDX] > 0.01)
        pos_hist_mask = pos_label_mask & (metrics_per_sample[METRICS_PRED_NDX] < 0.99)

        if neg_hist_mask.any():
            writer.add_histogram(
                'is_neg',
                metrics_per_sample[METRICS_PRED_NDX, neg_hist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )
        if pos_hist_mask.any():
            writer.add_histogram(
                'is_pos',
                metrics_per_sample[METRICS_PRED_NDX, pos_hist_mask],
                self.totalTrainingSamples_count,
                bins=bins,
            )



# usual 'if-main' stanza
if __name__ == "__main__":
    LunaTrainingApp().main()