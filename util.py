from collections import namedtuple
import numpy as np 
import logging
import logconfig
import time 
import datetime


log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

CandidateInfoTuple = namedtuple("CandidateInfoTuple", 
                                "isNodule_bool, diameter_mm, series_uid, center_xyz")

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

    if backoff is None:
        backoff = 2
        while backoff ** 7 < iter_len:
            backoff *= 2

    assert backoff >= 2
    while print_ndx < start_ndx * backoff:
        print_ndx *= backoff

    log.warning("{} ----/{}, starting".format(
        desc_str,
        iter_len,
    ))
    start_ts = time.time()
    for (current_ndx, item) in enumerate(iter):
        yield (current_ndx, item)
        if current_ndx == print_ndx:
            # ... <1>
            duration_sec = ((time.time() - start_ts)
                            / (current_ndx - start_ndx + 1)
                            * (iter_len-start_ndx)
                            )

            done_dt = datetime.datetime.fromtimestamp(start_ts + duration_sec)
            done_td = datetime.timedelta(seconds=duration_sec)

            log.info("{} {:-4}/{}, done at {}, {}".format(
                desc_str,
                current_ndx,
                iter_len,
                str(done_dt).rsplit('.', 1)[0],
                str(done_td).rsplit('.', 1)[0],
            ))

            print_ndx *= backoff

        if current_ndx + 1 == start_ndx:
            start_ts = time.time()

    log.warning("{} ----/{}, done at {}".format(
        desc_str,
        iter_len,
        str(datetime.datetime.now()).rsplit('.', 1)[0],
    ))
