import numpy as np 
import glob
import functools
import SimpleITK as sitk
from util import *
from disk_caching import * 
from segment_dset import find_radius, get_candidate_info_dict



class CT: 
    def __init__(self, series_uid, usage = "classifier"):
        
        mhd_path = glob.glob(f"/kaggle/input/luna16/subset*/subset*/{series_uid}.mhd")[0] # since returns a list 
        ct_mhd = sitk.ReadImage(mhd_path) # this method consumes the .raw file implicitly  
        ct_arr = np.array(sitk.GetArrayFromImage(ct_mhd), dtype = np.float32).clip(-1000, 1000)
        self.series_uid = series_uid
        self.hu_arr = ct_arr
        self.origin_xyz = XYZ_tuple(*ct_mhd.GetOrigin())
        self.voxel_sizes = XYZ_tuple(*ct_mhd.GetSpacing())
        self.transform_mat = np.array(ct_mhd.GetDirection()).reshape(3, 3)
        self.mode = usage

        if usage == "segment":
            self.candidate_info_list = get_candidate_info_dict(DATASET_DIR_PATH,subsets_included=(0,))[self.series_uid] # all the candidate nodules of that subject
            self.positive_candidates = [candidate for candidate in self.candidate_info_list if candidate.is_nodule] 
            self.positive_masks = self.build_annotation_mask(self.positive_candidates)
            # this line of code extracts the indices of the CT volume which has at least one voxel assigned value 1
            self.positive_indices = (self.positive_masks.sum(axis=(1,2)).nonzero()[0].tolist())

    def build_annotation_mask(self, ps_candidates, threshold_hu = -700):
        segmentation_mask = np.zeros_like(self.hu_arr, dtype=bool)

        for ps_candidate in ps_candidates:
            center_irc = xyz2irc(
                ps_candidate.center_xyz,
                self.origin_xyz,
                self.voxel_sizes,
                self.transform_mat,
            )

            ci = int(center_irc.index)
            cr = int(center_irc.row)
            cc = int(center_irc.column) 

            index_radius = find_radius(ci, cr, cc, 'index', self.hu_arr, threshold_hu)
            row_radius = find_radius(ci, cr, cc, 'row', self.hu_arr, threshold_hu)
            col_radius = find_radius(ci, cr, cc, 'col', self.hu_arr, threshold_hu)

            segmentation_mask[ci - index_radius: ci + index_radius, 
                              cr - row_radius: cr + row_radius, 
                              cc - col_radius: cc + col_radius] = True

        mask = segmentation_mask & (self.hu_arr > threshold_hu) # for the sake of ensuring proper masking 
        # That’s going to clip off the corners of our algorithmically searched mask

        return mask

    def get_raw_candidate_nodule(self, xyz_center, irc_diameters):
        """
        This method extracts the nodule and a small amount of data from its surrounding 
        for limiting the scope of the problem for the classifier.  
        
        Parameters: 
            - xyz_center (XYZ_tuple): the center of the potential nodule expressed in the patient space.  
            - irc_diameters (IRC_tuple): the spread of the nodule around its center expressed in the voxel space.
                                   This constant and don't depend on the diameter of the nodule for consistency.  
        
        return: 
            - ct_chunk: the tiny volumentric surrounding of the suspicious nodule. 
            - icr_center: The transformed center coordinates. 
        """
        icr_center = xyz2irc(xyz_center, self.origin_xyz, self.voxel_sizes, self.transform_mat) 
        
        slicing_list = list()
        for i, i_center in enumerate(icr_center):
            start_ind = int(round(i_center - irc_diameters[i]/2))
            end_ind = int(start_ind + irc_diameters[i])
            
            # report if there is any issue 
            assert i_center >= 0 and i_center < self.hu_arr.shape[i], repr([self.series_uid, xyz_center, self.origin_xyz, self.voxel_sizes, icr_center, i])
            
            # safety checks 
            if start_ind < 0: 
                start_ind = 0 
                end_ind = int(irc_diameters[i])
            
            if end_ind > self.hu_arr.shape[i]:
                end_ind = self.hu_arr.shape[i]
                start_ind = int(end_ind - irc_diameters[i])
            

            slicing_list.append(slice(start_ind, end_ind))
        
        ct_chunks = self.hu_arr[tuple(slicing_list)]
        if self.mode == "segment":
            pos_chunks = self.positive_masks[tuple(slicing_list)]
            return pos_chunks, ct_chunks, icr_center 
        else:
            return ct_chunks, icr_center




@functools.lru_cache(maxsize = 1, typed = True) # this would be enough if we are sure that the nodules loading will occur in order 
# meaning that all the candidate nodules of specific subject is first extracted, then the second subject's nodules and so on. 
def get_ct(series_uid, usage):
    return CT(series_uid, usage)

raw_cache = getCache("cache_candidates")

@raw_cache.memoize(typed = True) # save on disk to avoid loading the same ct scan each time to extract specific nodule surrounding 
# (just upload it once and save it for further nodules extraction from the same subject)
def get_ct_raw_candidates(series_uid ,xyz_center, irc_diameters, usage = 'classifier'):
    ct = get_ct(series_uid, usage)
    if usage == 'segment':
        pos_chunk, ct_chunk, icr_center = ct.get_raw_candidate_nodule(xyz_center, irc_diameters)
        return pos_chunk, ct_chunk, icr_center
    else:
        ct_chunk, icr_center = ct.get_raw_candidate_nodule(xyz_center, irc_diameters)
        return ct_chunk, icr_center
    

# this way we would save the returned value from this function call 
# (later when i call the same function with the same parameter, the function would read it off the disk) 
@raw_cache.memoize(typed = True)
def getCtSampleSize(series_uid):
    ct = CT(series_uid, usage = "segment")
    return int(ct.hu_arr.shape[0]), ct.positive_indices
    # since the number of channels is different between CT scans



