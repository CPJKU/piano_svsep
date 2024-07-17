from collections import defaultdict
import os
import errno


def idx_tuple_to_dict(idx_tuple, datasets_map):
    """Transforms indices of a list of tuples of indices (dataset, piece_in_dataset)
    into a dict {dataset: [piece_in_dataset,...,piece_in_dataset]}"""
    result_dict = defaultdict(list)
    for x in idx_tuple:
        result_dict[datasets_map[x][0]].append(datasets_map[x][1])
    return result_dict

def idx_dict_to_tuple(idx_dict):
    result_tuples = list()
    for k in idx_dict.keys():
        for v in idx_dict[k]:
            result_tuples.append((k,v))
    return result_tuples

def get_download_dir():
    """Get the absolute path to the download directory.
    Returns
    -------
    dirname : str
        Path to the download directory
    """
    default_dir = os.path.join(os.path.expanduser('~'), '.piano_svsep_data')
    dirname = os.environ.get('STRUTTURA_DOWNLOAD_DIR', default_dir)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    return dirname


def makedirs(path):
    try:
        os.makedirs(os.path.expanduser(os.path.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and os.path.isdir(path):
            raise

