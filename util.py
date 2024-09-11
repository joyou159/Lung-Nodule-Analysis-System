from collections import namedtuple
import numpy as np 
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


CandidateInfoTuple = namedtuple("CandidateInfoTuple", 
                                "isNodule_bool, diameter_mm, series_uid, center_xyz")


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