import functools
import glob
import os
import csv
from collections import namedtuple


CandidateSegInfo = namedtuple('CandidateSegInfo', 'is_nodule, hasAnnotation_bool, isMal_bool, diameter_mm, series_uid, center_xyz')


@functools.lru_cache(maxsize=1)  # caches the results of function call, if it have been called with the same argument. 
def get_candidate_info_list(dataset_dir_path, required_on_desk=True, subsets_included = (0,1,2,3,4)):

    mhd_list = glob.glob("/kaggle/input/luna16/subset*/subset*/*.mhd") # extract all 
    if required_on_desk:
        filtered_mhd_list = [mhd for mhd in mhd_list if any(f"subset{id}" in mhd for id in subsets_included)]
        uids_present_on_disk = {os.path.split(p)[-1][:-4] for p in filtered_mhd_list} # the unique series_uids for further filtration
        mhd_list = uids_present_on_disk

    candidates_list = list()

    # extract the nodules (whether they are malignant or benign) that has annotations data without repetition
    with open(os.path.join(dataset_dir_path, "annotations_for_segmentation.csv"), "r") as f:
        for row in list(csv.reader(f))[1:]: 
            series_uid = row[0]
            if series_uid not in mhd_list:
                continue

            center_xyz = tuple([float(x) for x in row[1:4]])
            diameter = float(row[4])
            is_malignant = {"False": False, "True": True}[row[5]] 


            candidates_list.append(
                CandidateSegInfo(
                    True, # it's a nodule 
                    True, # has annotations data already (meaning that there are some nodules that have no annotation data)
                    is_malignant,
                    diameter,
                    series_uid, 
                    center_xyz
                )
            )
    
    # extract non-nodule (negative examples so that the U-net model learns to ignore them)
    with open(os.path.join(dataset_dir_path, "annotations.csv"), "r") as f:
        for row in list(csv.reader(f))[1:]: 
            series_uid = row[0]

            if series_uid not in mhd_list:
                continue

            is_nodule = bool(int(row[4]))
            center_xyz = tuple([float(x) for x in row[1:4]])

            if not is_nodule: 
                candidates_list.append(
                    CandidateSegInfo(
                        False,   
                        False,
                        False,
                        diameter,
                        series_uid, 
                        center_xyz
                    )
                )

    return candidates_list


def find_radius(ci, cr, cc, axis, hu_arr, threshold_hu):
    """
    For mask generation 
    """
    radius = 2
    try:
        while True:
            # Check based on the axis
            if axis == 'index':
                if hu_arr[ci + radius, cr, cc] <= threshold_hu or hu_arr[ci - radius, cr, cc] <= threshold_hu:
                    break
            elif axis == 'row':
                if hu_arr[ci, cr + radius, cc] <= threshold_hu or hu_arr[ci, cr - radius, cc] <= threshold_hu:
                    break
            elif axis == 'col':
                if hu_arr[ci, cr, cc + radius] <= threshold_hu or hu_arr[ci, cr, cc - radius] <= threshold_hu:
                    break
            # Increment the radius if the condition is met
            radius += 1
    except IndexError:
        radius -= 1  # Fix the last incorrect incrementation due to out-of-bounds access
    return radius

@functools.lru_cache(1)
def get_candidate_info_dict(dataset_dir_path, required_on_desk=True, subsets_included = (0,)):
    candidate_list = get_candidate_info_list(dataset_dir_path, required_on_desk, subsets_included)
    candidate_dict = dict()

    for candidate_tuple in candidate_list:
        candidate_dict.setdefault(candidate_tuple.series_uid, []).append(candidate_tuple)
    
    return candidate_dict

