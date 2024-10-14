import argparse
import sys
import datetime
import torch 
import torch.nn as nn 
from torch.utils.data import DataLoader
from common_utils.util import *
from common_utils.logconfig import *
from classifier_dset import *
from NoduleClassifier import NoduleClassifier 
from torch.utils.tensorboard import SummaryWriter
import hashlib
import shutil
import matplotlib.pyplot as plt

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

METRICS_LABEL_NDX=0
METRICS_PRED_NDX=1
METRICS_PRED_P_NDX=2
METRICS_LOSS_NDX=3
METRICS_SIZE = 4

class ClassificationTrainingApp():
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument("--num-workers", 
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
        
        parser.add_argument('--dataset',
            help="What to dataset to feed the model.",
            action='store',
            default='LunaDataset', # later we would pass the malignancy/benign dataset. 
        )
        
        parser.add_argument('--malignant',
            help="Train the model to classify nodules as benign or malignant.",
            action='store_true',
            default=False, # meaning Nodule classificaiton task 
        )
        parser.add_argument('--finetune',
            help="Start finetuning from this model.",
            default='',
        )
        parser.add_argument('--finetune-depth',
            help="Number of blocks (counted from the head) to include in finetuning",
            type=int,
            default=1,
        )
        parser.add_argument('--tb-prefix',
            default='Nodule',
            help="Data prefix to use for Tensorboard run.",
        )
        parser.add_argument('comment',
            help="Comment suffix for Tensorboard run.",
            nargs='?',
            default='luna',
        )


        self.args_list  = parser.parse_args(sys_argv)
        self.time_str = datetime.datetime.now().strftime('%Y-%m-%d_%H.%M.%S') # to identify running time
        
        self.train_writer = None
        self.val_writer = None
        self.totalTrainingSamples_count = 0
        
        self.augmentation_dict = {}
        if True: # default 
            self.augmentation_dict['flip'] = True
            self.augmentation_dict['offset'] = 0.1
            self.augmentation_dict['scale'] = 0.5
            self.augmentation_dict['rotate'] = True
            self.augmentation_dict['noise'] = 0 

        self.use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")

        self.model = self.init_model()
        self.optimizer = self.init_optimizer()


    def init_model(self):
        model = NoduleClassifier()

        if self.args_list.finetune:
            d = torch.load(self.args_list.finetune, map_location='cpu')
            model_blocks = [
                n for n, subm in model.named_children()
                if len(list(subm.parameters())) > 0
            ]
            finetune_blocks = model_blocks[-self.args_list.finetune_depth:]
            log.info(f"finetuning from {self.args_list.finetune}, blocks {' '.join(finetune_blocks)}")
            model.load_state_dict(
                {
                    k: v for k,v in d['model_state'].items()
                    if k.split('.')[0] not in model_blocks[-1]
                },
                strict=False,
            )
            for n, p in model.named_parameters():
                if n.split('.')[0] not in finetune_blocks:
                    p.requires_grad_(False)
        if self.use_cuda:
            log.info("Using CUDA; {} devices.".format(torch.cuda.device_count()))
            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            model = model.to(self.device)
        return model

    def init_optimizer(self):
        lr = 0.003 if self.args_list.finetune else 0.001
        return torch.optim.SGD(self.model.parameters(), lr=lr, weight_decay=1e-4)

    def init_data_loader(self, val_set_bool = False):
        if self.args_list.dataset == "LunaDataset":
            dataset = LunaDataset
        else:
            dataset = MalignantLunaDataset
        
        
        if val_set_bool: # no balancing for validation (real-world isn't balanced anyway)
            dataset = dataset(DATASET_DIR_PATH, 
                                  self.args_list.subsets_included, 
                                  val_set_bool=val_set_bool,
                                  val_stride=10)
        else: # ratio_int = 1 (alternating)
            dataset = dataset(DATASET_DIR_PATH,
                                 self.args_list.subsets_included,
                                 val_set_bool=val_set_bool,
                                 val_stride=10,
                                 ratio_int=1)
            
        batch_size = self.args_list.batch_size

        if self.use_cuda:
            batch_size *= torch.cuda.device_count()  # each GPU has its own batch 
        
        data_loader = DataLoader(dataset, batch_size=batch_size,
                               num_workers=self.args_list.num_workers  # this refers to the num of CPU processes to load data to memory in parallel 
                               ,pin_memory=self.use_cuda) # transfer the data in the memory to the GPU quickly

        return data_loader

    def initTensorboardWriters(self):
        if self.train_writer is None:
            log_dir = os.path.join('Classification', self.args_list.tb_prefix,
                                   self.time_str)

            self.train_writer = SummaryWriter(
                log_dir=log_dir + '-train_cls-' + self.args_list.comment)
            self.val_writer = SummaryWriter(
                log_dir=log_dir + '-val_cls-' + self.args_list.comment)


    def main(self):
        log.info("Starting {}, {}".format(type(self).__name__, self.args_list))

        train_dl = self.init_data_loader(val_set_bool = False)
        val_dl = self.init_data_loader(val_set_bool = True)

        best_score = 0.0
        validation_cadence = 5 if not self.args_list.finetune else 1
        for epoch_ndx in range(1, self.args_list.epochs + 1):
            
            log.info(f"epoch no.{epoch_ndx} of {self.args_list.epochs} -- (train_dl/val_dl): {len(train_dl)}/{len(val_dl)} -- with batch size of {self.args_list.batch_size} on {torch.cuda.device_count()} GPUs")

            train_metrics_per_sample = self.training_epoch(epoch_ndx, train_dl)
            self.log_metrics(epoch_ndx, 'train', train_metrics_per_sample)

            if epoch_ndx == 1 or epoch_ndx % validation_cadence == 0:
                val_metrics_per_sample = self.validation_epoch(epoch_ndx, val_dl)
                score = self.log_metrics(epoch_ndx, 'val', val_metrics_per_sample)
                best_score = max(score, best_score)

                self.save_model('cls', epoch_ndx, score == best_score)

        self.train_writer.close()
        self.val_writer.close()


    def training_epoch(self, epoch_ndx, train_dl):
        self.model.train()
        train_dl.dataset.shuffle_samples()
        
        train_metrics_per_sample = torch.zeros(
            METRICS_SIZE,
            len(train_dl.dataset),
            device=self.device,
        )

        batch_iter = enumerateWithEstimate(
            train_dl,
            f"E{epoch_ndx} Training",
            start_ndx=train_dl.num_workers)
        
        for batch_ndx, batch_tup in batch_iter:
            self.optimizer.zero_grad()

            loss_var = self.compute_batch_loss(
                batch_ndx,
                batch_tup,
                train_dl.batch_size,
                train_metrics_per_sample,
                augment=True
            )

            loss_var.backward()
            self.optimizer.step()

        self.totalTrainingSamples_count += len(train_dl.dataset)

        return train_metrics_per_sample.to('cpu')
    
    
    def validation_epoch(self, epoch_ndx, val_dl):
        with torch.no_grad():
            self.model.eval()
            val_metrics_per_sample = torch.zeros(
                METRICS_SIZE,
                len(val_dl.dataset),
                device=self.device,
            )

            batch_iter = enumerateWithEstimate(
            val_dl,
            f"E{epoch_ndx} Validation",
            start_ndx=val_dl.num_workers)

            for batch_ndx, batch_tup in batch_iter:
                self.compute_batch_loss(
                    batch_ndx,
                    batch_tup,
                    val_dl.batch_size,
                    val_metrics_per_sample,
                    augment=False
                )

        return val_metrics_per_sample.to('cpu')

    def compute_batch_loss(self, batch_ndx, batch_tup, batch_size, metrics_g,
                         augment=True):
        input_t, label_t, index_t, _series_list, _center_list = batch_tup

        input_g = input_t.to(self.device, non_blocking=True)
        label_g = label_t.to(self.device, non_blocking=True)
        index_g = index_t.to(self.device, non_blocking=True)

        if augment:
            input_g = augmentation_3D(input_g, self.augmentation_dict)

        logits_g, probability_g = self.model(input_g)

        loss_g = nn.functional.cross_entropy(logits_g, label_g[:, 1],
                                             reduction="none")
        start_ndx = batch_ndx * batch_size
        end_ndx = start_ndx + label_t.size(0)

        _, predLabel_g = torch.max(probability_g, dim=1, keepdim=False,
                                   out=None)

        metrics_g[METRICS_LABEL_NDX, start_ndx:end_ndx] = index_g
        metrics_g[METRICS_PRED_NDX, start_ndx:end_ndx] = predLabel_g
        metrics_g[METRICS_PRED_P_NDX, start_ndx:end_ndx] = probability_g[:,1]
        metrics_g[METRICS_LOSS_NDX, start_ndx:end_ndx] = loss_g

        return loss_g.mean()


    def log_metrics(
            self,
            epoch_ndx,
            mode_str,
            metrics_t,
            classificationThreshold=0.5,
    ):
        self.initTensorboardWriters()
        log.info("E{} {}".format(
            epoch_ndx,
            type(self).__name__,
        ))

        if self.args_list.dataset == 'MalignantLunaDataset':
            pos = 'mal'
            neg = 'ben'
        else:
            pos = 'pos'
            neg = 'neg'


        negLabel_mask = metrics_t[METRICS_LABEL_NDX] == 0
        negPred_mask = metrics_t[METRICS_PRED_NDX] == 0

        posLabel_mask = ~negLabel_mask
        posPred_mask = ~negPred_mask


        neg_count = int(negLabel_mask.sum())
        pos_count = int(posLabel_mask.sum())

        neg_correct = int((negLabel_mask & negPred_mask).sum())
        pos_correct = int((posLabel_mask & posPred_mask).sum())

        trueNeg_count = neg_correct
        truePos_count = pos_correct

        falsePos_count = neg_count - neg_correct
        falseNeg_count = pos_count - pos_correct

        metrics_dict = {}
        metrics_dict['loss/all'] = metrics_t[METRICS_LOSS_NDX].mean()
        metrics_dict['loss/neg'] = metrics_t[METRICS_LOSS_NDX, negLabel_mask].mean()
        metrics_dict['loss/pos'] = metrics_t[METRICS_LOSS_NDX, posLabel_mask].mean()

        metrics_dict['correct/all'] = (pos_correct + neg_correct) / metrics_t.shape[1] * 100
        metrics_dict['correct/neg'] = (neg_correct) / neg_count * 100
        metrics_dict['correct/pos'] = (pos_correct) / pos_count * 100

        precision = metrics_dict['pr/precision'] = \
            truePos_count / np.float64(truePos_count + falsePos_count)
        recall    = metrics_dict['pr/recall'] = \
            truePos_count / np.float64(truePos_count + falseNeg_count)

        metrics_dict['pr/f1_score'] = \
            2 * (precision * recall) / (precision + recall)

        threshold = torch.linspace(1, 0, steps=100)  
        tpr = (metrics_t[None, METRICS_PRED_P_NDX, posLabel_mask] >= threshold[:, None]).sum(1).float() / pos_count
        fpr = (metrics_t[None, METRICS_PRED_P_NDX, negLabel_mask] >= threshold[:, None]).sum(1).float() / neg_count
        fp_diff = fpr[1:]-fpr[:-1]
        tp_avg  = (tpr[1:]+tpr[:-1])/2
        auc = (fp_diff * tp_avg).sum()
        metrics_dict['auc'] = auc

        log.info(
            ("E{} {:8} {loss/all:.4f} loss, "
                 + "{correct/all:-5.1f}% correct, "
                 + "{pr/precision:.4f} precision, "
                 + "{pr/recall:.4f} recall, "
                 + "{pr/f1_score:.4f} f1 score, "
                 + "{auc:.4f} auc"
            ).format(
                epoch_ndx,
                mode_str,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/neg:.4f} loss, "
                 + "{correct/neg:-5.1f}% correct ({neg_correct:} of {neg_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + neg,
                neg_correct=neg_correct,
                neg_count=neg_count,
                **metrics_dict,
            )
        )
        log.info(
            ("E{} {:8} {loss/pos:.4f} loss, "
                 + "{correct/pos:-5.1f}% correct ({pos_correct:} of {pos_count:})"
            ).format(
                epoch_ndx,
                mode_str + '_' + pos,
                pos_correct=pos_correct,
                pos_count=pos_count,
                **metrics_dict,
            )
        )

        writer = getattr(self, mode_str + '_writer')

        for key, value in metrics_dict.items():
            key = key.replace('pos', pos)
            key = key.replace('neg', neg)
            writer.add_scalar(key, value, self.totalTrainingSamples_count)

        fig = plt.figure()
        plt.plot(fpr, tpr)
        writer.add_figure('roc', fig, self.totalTrainingSamples_count)

        writer.add_scalar('auc', auc, self.totalTrainingSamples_count)

        bins = np.linspace(0, 1, num=100)

        writer.add_histogram(
            'label_neg',
            metrics_t[METRICS_PRED_P_NDX, negLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )
        writer.add_histogram(
            'label_pos',
            metrics_t[METRICS_PRED_P_NDX, posLabel_mask],
            self.totalTrainingSamples_count,
            bins=bins
        )

        if not self.args_list.malignant:
            score = metrics_dict['pr/f1_score']
        else:
            score = metrics_dict['auc']

        return score

    def save_model(self, type_str, epoch_ndx, isBest=False):
        file_path = os.path.join(
            'models',
            self.args_list.tb_prefix,
            '{}_{}_{}.{}.state'.format(
                type_str,
                self.time_str,
                self.args_list.comment,
                self.totalTrainingSamples_count,
            )
        )

        os.makedirs(os.path.dirname(file_path), mode=0o755, exist_ok=True)

        model = self.model
        if isinstance(model, torch.nn.DataParallel):
            model = model.module

        state = {
            'model_state': model.state_dict(),
            'model_name': type(model).__name__,
            'optimizer_state' : self.optimizer.state_dict(),
            'optimizer_name': type(self.optimizer).__name__,
            'epoch': epoch_ndx,
            'totalTrainingSamples_count': self.totalTrainingSamples_count,
        }
        torch.save(state, file_path)

        log.debug("Saved model params to {}".format(file_path))

        if isBest:
            best_path = os.path.join(
                'models',
                self.args_list.tb_prefix,
                '{}_{}_{}.{}.state'.format(
                    type_str,
                    self.time_str,
                    self.args_list.comment,
                    'best',
                )
            )
            shutil.copyfile(file_path, best_path)

            log.debug("Saved model params to {}".format(best_path))

        with open(file_path, 'rb') as f:
            log.info("SHA1: " + hashlib.sha1(f.read()).hexdigest())

# usual 'if-main' stanza
if __name__ == "__main__":
    ClassificationTrainingApp().main()