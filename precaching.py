from util import *
from dsets import LunaDataset
from torch.utils.data import DataLoader
import argparse
import sys

""" This App is used for creating the cache_candidates as on-disk caching to 
    speed up the training and validation epochs.
"""
class LunaPrepCacheApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size',
            help='Batch size to use for training',
            default=64,
            type=int,
        )
        parser.add_argument('--num_workers',
            help='Number of worker processes for background data loading',
            default=8,
            type=int,
        )
        
        parser.add_argument('--subsets_included',
            help='The number of subsets included in the training process',
            default=(0,1,2,3,4),
            type=tuple,
        )

        self.args_list = parser.parse_args(sys_argv)

    def main(self):
        print("Starting {}, {}".format(type(self).__name__, self.args_list))

        self.prep_dl = DataLoader(
            LunaDataset(DATASET_DIR_PATH, self.args_list.subsets_included,
                sortby_str='series_uid',
            ),
            batch_size=self.args_list.batch_size,
            num_workers=self.args_list.num_workers,
        )

        for ndx,_ in enumerate(self.prep_dl):
            print(f"Batch {ndx} out of {len(self.prep_dl)} -> Pre-Caching")
