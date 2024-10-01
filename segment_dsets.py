import functools

def get_candidate_info_list(dataset_dir_path, required_on_desk=True, subsets_included = (0,1,2,3,4)):
    pass


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

