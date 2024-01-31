"""
    Utility functions for preprocessing vim-1 dataset.
    Date created: 18/02/19
    Python Version: 3.6
"""

#from GPUtils.startup_guyga import *
from numpy import * #for compatibility with Roman's code
import h5py
import os
import itertools
from itertools import chain, permutations, combinations, product
from scipy.io import loadmat



dataset_root = '/net/mraid11/export/groups/iranig/datasets/vim-1/'
n_repeat = [2, 13]  # train, test
n_TR_trial = 1

# According to indexing
roi_labels = ['other', 'V1', 'V2', 'V3', 'V3A', 'V3B', 'V4', 'LO']

def listify(value):
    """ Ensures that the value is a list. If it is not a list, it creates a new list with `value` as an item. """
    if not isinstance(value, list):
        value = [value]
    return value


def get_aux(subject_idx):
    '''
    subject_idx should be 1, 2, or 3.
    Return:
    'voxIdx' - indices into the 86 x 86 x 34 functional volume
    'roi' - ROI assignment (index)
    'snr' - SNR value, calculated from the training data as absolute value
            of the beta weight divided by the standard error of the beta weight.
            for each voxel, a single SNR number is reported --- this number is the
            median SNR value observed across the 1750 training images.
    '''
    aux = loadmat(os.path.join(dataset_root, 'S%daux.mat' % subject_idx))
    return { 'voxIdx': aux['voxIdxS%d' % subject_idx],
             'roi': aux['roiS%d' % subject_idx],
             'snr': aux['snrS%d' % subject_idx] }


def get_roi_label(subject_indices=[1, 2, 3, 4, 5]):
    return {sbj_idx: vectorize(lambda i: roi_labels[i])(get_aux(sbj_idx)['roi']) for sbj_idx in listify(subject_indices)}

def get_processed_data(sbj_idx, avg=False):
    '''
    avg means get data that was averaged and then normalized. Otherwise get single trial data which is averaged independently.
    Returns: train_data, test_data, train_lbl, test_lbl
    '''
    data = []
    labels = []
    if avg:
        file = loadmat(os.path.join(dataset_root, 'S{}data.mat'.format(sbj_idx)))
        for cat_idx, cat_str in enumerate(['Trn', 'Val']):
            data.append(file['data{}S{}'.format(cat_str, sbj_idx)].T)
            # Labels are trivial
            labels.append(arange(len(data[-1])))
    else:
        for cat_idx, cat_str in enumerate(['Trn', 'Val']):
            file = h5py.File(os.path.join(dataset_root, 'S{}data_{}_singletrial.mat'.format(sbj_idx, cat_str.lower())), 'r')
            data.append(array(file['data{}SingleS{}'.format(cat_str, sbj_idx)]))
            N = int(len(data[-1]) // n_repeat[cat_idx])
            labels.append(tile(arange(N), (n_repeat[cat_idx], 1)).flatten())

    return list(itertools.chain(data, labels))

# get_processed_data(1)
# get_aux(1)
# get_roi_label(1)
