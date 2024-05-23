# -*- coding: utf-8 -*-
"""
Utility functions for Yggdrasil
"""
from os.path import join
import pandas as pd
import numpy as np
import re


def get_starts(file_name, ecephys_path, gate=0):
    '''Get start time (sec) of each gate in concatenated spikeGLX files'''
    animal = file_name.split('/')[0]
    rec_file_stem = file_name.split('/')[1]
    rec_file_stem = rec_file_stem[:-3]
    ni_file_path = join(ecephys_path, animal, 'Ecephys',
                        rec_file_stem, 'catgt_'+rec_file_stem+'_g'+repr(gate))
    ap_file_path = join(ni_file_path, rec_file_stem+'_g'+repr(gate)+'_imec0')

    # get start & end offset time of epochs
    offsets_file = join(ni_file_path, rec_file_stem+'_g'+repr(gate)+'_ct_offsets.txt')
    offsets = pd.read_csv(offsets_file, sep='\t', header=None, index_col=0)
    offsets = offsets.T

    if 'sec_nidq:' in offsets.columns:
        starts = offsets['sec_nidq:'].copy()
    elif 'sec_imlf0:' in offsets.columns:
        starts = offsets['sec_imlf0:'].copy()
    elif 'sec_imap0:' in offsets.columns:
        starts = offsets['sec_imap0:'].copy()
    elif 'sec_imap1:' in offsets.columns:
        starts = offsets['sec_imap1:'].copy()
    else:
        raise Exception('Could not find start/stop time in seconds')

    # get end time of recording
    ap_meta_file = join(ap_file_path, rec_file_stem+'_g'+repr(gate)+'_tcat.imec0.ap.meta')
    #ap_meta_file = join(ni_file_path, rec_file_stem+'_g'+repr(gate)+'_tcat.nidq.meta')
    file_stream = open(ap_meta_file)
    for line in file_stream:
        if 'fileTimeSecs' in line:
            rec_end = float(re.findall(r'(\d+\.\d+)', line)[0])
    starts = pd.concat([starts, pd.Series(rec_end)])
    starts = starts.reset_index(drop=True)

    return starts


def subset_intervals_by_length(intervals, min_length=0.075):
    '''Find intervals of a certain length (sec) or greater
    Run after get_cycle_intervals to pick intervals containing an entire theta cycle'''
    subset_intervals = intervals
    for i, interval in enumerate(intervals):
        if interval[1]-interval[0] < min_length:
            subset_intervals.pop(i)
    return subset_intervals


def convert_times_to_feature(feature, feature_times, query_times):
    '''Given a 1xn list of (spike)times, map to closest feature at those times
    E.g. convert_times_to_feature(position.position['Linear'], 
                                  position.postiion['Timestamp'],
                                  spikes.spikes[0])
    would return the linearized position at each spike'''
    feature_list = np.empty_like(query_times)
    for i, t in enumerate(query_times):
        idx = np.argmin(np.abs(feature_times - t))
        feature_list[i] = feature[idx]
    return feature_list