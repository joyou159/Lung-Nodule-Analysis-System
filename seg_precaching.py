from seg_dset_utils import get_candidate_info_list
from torch.utils.data import Dataset 
import sys
import argparse
from torch.utils.data import DataLoader
from CT import get_ct_raw_candidates, get_ct_sample_size
from logconfig import *
from util import enumerateWithEstimate, DATASET_DIR_PATH

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class PrepcacheSegDataset(Dataset):
    def __init__(self, dataset_dir_path, subsets_included):
        super().__init__()
        self.subsets_included = subsets_included
        self.candidateInfo_list = get_candidate_info_list(dataset_dir_path, subsets_included = subsets_included)
        self.pos_list = [nt for nt in self.candidateInfo_list if nt.is_nodule]

        self.seen_set = set()
        self.candidateInfo_list.sort(key=lambda x: x.series_uid)

    def __len__(self):
        return len(self.candidateInfo_list)

    def __getitem__(self, ndx):
        
        candidateInfo_tup = self.candidateInfo_list[ndx]
        # this function call caches candidates on the disk. 
        get_ct_raw_candidates(candidateInfo_tup.series_uid, candidateInfo_tup.center_xyz, (7, 96, 96), self.subsets_included, usage = "segment")

        series_uid = candidateInfo_tup.series_uid
        if series_uid not in self.seen_set:
            self.seen_set.add(series_uid)

            get_ct_sample_size(series_uid, self.subsets_included)

        return 0 


class SegPrepCacheApp:
    def __init__(self, sys_argv=None):
        if sys_argv is None:
            sys_argv = sys.argv[1:]

        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', help='Batch size to use for training', default=1024, type=int)
        parser.add_argument('--num_workers', help='Number of worker processes for background data loading', default=8, type=int)
        parser.add_argument('--subsets_included', help='The number of subsets included in the training process', default=(0, 1, 2, 3, 4), type=tuple)
        
        # Parse arguments
        self.arg_list = parser.parse_args(sys_argv)

    def main(self):
        log.info(f"Starting {type(self).__name__}, with args: {self.arg_list}")

        # Example DataLoader initialization (change as per your dataset)
        self.prep_dl = DataLoader(
            PrepcacheSegDataset(DATASET_DIR_PATH, self.arg_list.subsets_included),  # Assuming PrepcacheSegDataset is defined
            batch_size=self.arg_list.batch_size,
            num_workers=self.arg_list.num_workers,
        )

        batch_iter = enumerateWithEstimate(
            self.prep_dl,
            "Stuffing cache",
            start_ndx=self.prep_dl.num_workers,
        )
        for batch_ndx, batch_tup in batch_iter:
            pass

if __name__ == '__main__':
    SegPrepCacheApp().main()