from collections import namedtuple
import numpy as np 
import logging
import logconfig
import time 
import datetime
import csv 
import glob 
import functools
import os


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

Candidate_Info = namedtuple("Candidate_Info", 
                                "isNodule_bool, diameter_mm, series_uid, center_xyz")

CandidateInfoTuple = namedtuple('CandidateInfoTuple', 'is_nodule, has_annotations, is_malignant, diameter_mm, series_uid, center_xyz')

AugmentationInfo = namedtuple("AugmentationInfo", 
                                "flip, offset, scale, rotate, noise")


DATASET_DIR_PATH = "luna/"

# for consistency 
IRC_tuple = namedtuple('IRC_tuple', ["index", "row", "column"]) 
XYZ_tuple = namedtuple('XYZ_tuple', ["x", "y", "z"]) 

def irc2xyz(irc_coord, xyz_origin, xyz_sizes, irc_transform_mat):
    """
    Map the voxel-based coordinates to the patient coordidates through data included in the meta-data file of CT.

    Parameters:
        - irc_coord (IRC_tuple): the voxel coordinates to be transformed (index, row, column).
        - xyz_origin (XYZ_tuple): the exact origin point in the patient space for reference. 
        - xyz_sizes (XYZ_tuple): the voxel size for scaling purposes. 
        - irc_transform_mat (np.array): the transformation matrix between the two spaces.

    return:
        transformed coordinate as XYZ_tuple
    """
    cri_coord = np.array(irc_coord)[::-1]
    physical_origin = np.array(xyz_origin)
    physical_sizes = np.array(xyz_sizes)
    xyz_coord = irc_transform_mat @ (cri_coord * physical_sizes) + physical_origin 
    return XYZ_tuple(*xyz_coord)

def xyz2irc(xyz_coord, xyz_origin, xyz_sizes, irc_transform_mat):
    """
    Map the patient coordidates to the voxel-based coordinates through data included in the meta-data file of CT.

    Parameters:
        - xyz_coord (XYZ_tuple): the patient coordinates to be transformed.
        - xyz_origin (XYZ_tuple): the exact origin point in the patient space for reference. 
        - xyz_sizes (XYZ_tuple): the voxel size for scaling purposes. 
        - irc_transform_mat (np.array): the transformation matrix between the two spaces.

    return:
        transformed coordinate as IRC_tuple
    """
    coordinate_xyz = np.array(xyz_coord) 
    physical_origin = np.array(xyz_origin)
    physical_sizes = np.array(xyz_sizes)
    # reverse the computations
    cri_coord = ((coordinate_xyz - physical_origin) @ np.linalg.inv(irc_transform_mat)) / physical_sizes
    rounded_cri_coord = np.round(cri_coord).astype(int)     
    return IRC_tuple(*rounded_cri_coord[::-1])

import functools
import os
import glob
import csv

@functools.lru_cache(maxsize=1)
def get_candidate_info_list(dataset_dir_path, required_on_desk=True, subsets_included=(0, 1, 2, 3, 4)):
    """
    Retrieves a list of candidate nodules from the dataset, including annotated nodules and non-nodules with malignancy info included.

    Args:
        dataset_dir_path (str): The path to the dataset directory containing the candidate information.
        required_on_desk (bool, optional): If True, filters the files based on whether they are in the specified subsets. Defaults to True.
        subsets_included (tuple of int, optional): Subsets to include for filtering if required_on_desk is True. Defaults to (0, 1, 2, 3, 4).

    Returns:
        list: A list of CandidateInfoTuple objects containing information about nodules and non-nodules.
    """
    mhd_list = glob.glob("/kaggle/input/luna16/subset*/subset*/*.mhd")
    if required_on_desk:
        filtered_mhd_list = [mhd for mhd in mhd_list if any(f"subset{id}" in mhd for id in subsets_included)]
        uids_present_on_disk = {os.path.split(p)[-1][:-4] for p in filtered_mhd_list}
        mhd_list = uids_present_on_disk

    candidates_list = []

    # Extract nodules with annotations data
    with open("/kaggle/input/annotations-for-segmentation/annotations_for_malignancy.csv", "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in mhd_list:
                continue

            center_xyz = tuple([float(x) for x in row[1:4]])
            diameter = float(row[4])
            is_malignant = {"False": False, "True": True}[row[5]]

            candidates_list.append(
                CandidateInfoTuple(
                    True,  # It's a nodule
                    True,  # Has annotation data
                    is_malignant,
                    diameter,
                    series_uid,
                    center_xyz
                )
            )

    # Extract non-nodules (negative examples)
    with open(os.path.join(dataset_dir_path, "candidates.csv"), "r") as f:
        for row in list(csv.reader(f))[1:]:
            series_uid = row[0]
            if series_uid not in mhd_list:
                continue

            is_nodule = bool(int(row[4]))
            center_xyz = tuple([float(x) for x in row[1:4]])

            if not is_nodule:
                candidates_list.append(
                    CandidateInfoTuple(
                        False,
                        False,
                        False,
                        0.0,
                        series_uid,
                        center_xyz
                    )
                )
    candidates_list.sort(reverse=True)

    return candidates_list

@functools.lru_cache(maxsize=1)
def get_candidate_info_dict(dataset_dir_path, required_on_desk=True, subsets_included=(0, 1, 2, 3, 4)):
    """
    Retrieves a dictionary mapping series UIDs to lists of candidate nodules.

    Args:
        dataset_dir_path (str): The path to the dataset directory containing the candidate information.
        required_on_desk (bool, optional): If True, filters the files based on whether they are in the specified subsets. Defaults to True.
        subsets_included (tuple of int, optional): Subsets to include for filtering if required_on_desk is True. Defaults to (0, 1, 2, 3, 4).

    Returns:
        dict: A dictionary where each key is a series UID, and the value is a list of CandidateInfoTuple objects.
    """
    candidate_list = get_candidate_info_list(dataset_dir_path, required_on_desk, subsets_included)
    candidate_dict = {}

    for candidate_tuple in candidate_list:
        candidate_dict.setdefault(candidate_tuple.series_uid, []).append(candidate_tuple)

    return candidate_dict


"""
    Source: https://github.com/deep-learning-with-pytorch/dlwpt-code/blob/master/util/util.py
"""

def enumerateWithEstimate(
        iter,
        desc_str,
        start_ndx=0,
        print_ndx=4,
        backoff=None,
        iter_len=None,
):
    """
    In terms of behavior, `enumerateWithEstimate` is almost identical
    to the standard `enumerate` (the differences are things like how
    our function returns a generator, while `enumerate` returns a
    specialized `<enumerate object at 0x...>`).

    However, the side effects (logging, specifically) are what make the
    function interesting.

    :param iter: `iter` is the iterable that will be passed into
        `enumerate`. Required.

    :param desc_str: This is a human-readable string that describes
        what the loop is doing. The value is arbitrary, but should be
        kept reasonably short. Things like `"epoch 4 training"` or
        `"deleting temp files"` or similar would all make sense.

    :param start_ndx: This parameter defines how many iterations of the
        loop should be skipped before timing actually starts. Skipping
        a few iterations can be useful if there are startup costs like
        caching that are only paid early on, resulting in a skewed
        average when those early iterations dominate the average time
        per iteration.

        NOTE: Using `start_ndx` to skip some iterations makes the time
        spent performing those iterations not be included in the
        displayed duration. Please account for this if you use the
        displayed duration for anything formal.

        This parameter defaults to `0`.

    :param print_ndx: determines which loop interation that the timing
        logging will start on. The intent is that we don't start
        logging until we've given the loop a few iterations to let the
        average time-per-iteration a chance to stablize a bit. We
        require that `print_ndx` not be less than `start_ndx` times
        `backoff`, since `start_ndx` greater than `0` implies that the
        early N iterations are unstable from a timing perspective.

        `print_ndx` defaults to `4`.

    :param backoff: This is used to how many iterations to skip before
        logging again. Frequent logging is less interesting later on,
        so by default we double the gap between logging messages each
        time after the first.

        `backoff` defaults to `2` unless iter_len is > 1000, in which
        case it defaults to `4`.

    :param iter_len: Since we need to know the number of items to
        estimate when the loop will finish, that can be provided by
        passing in a value for `iter_len`. If a value isn't provided,
        then it will be set by using the value of `len(iter)`.

    :return:
    """
    if iter_len is None:
        iter_len = len(iter)

    if backoff is None: # backoff controls the rate at which logging occurs 
        backoff = 2
        while backoff ** 7 < iter_len: # finding a suitable rate  
            backoff *= 2

    assert backoff >= 2 
    while print_ndx < start_ndx * backoff: # 'print_ndx' specifies what is the upcoming index to be printed.
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))

    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item) # generator return 
        if current_ndx == print_ndx: 
            # ... <1>
            duration_sec = ((time.time() - start_ts) / (current_ndx - start_ndx + 1)) * (iter_len-start_ndx) # Estimates the remaining time

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0], # date & time get up to seconds (exclude microseconds)
                str(done_td).rsplit('.', 1)[0], # elapsed time get up to seconds (exclude microseconds)
            ))

            print_ndx *= backoff # update the upcoming print_ndx using the selected rate 

        if current_ndx + 1 == start_ndx: 
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))
