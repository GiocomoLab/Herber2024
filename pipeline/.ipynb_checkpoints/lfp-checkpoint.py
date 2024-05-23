# -*- coding: utf-8 -*-
'''
Class to read LFP voltage trace into Yggdrasil format
Use:
    (1) call init with all parameters defined except 'name' to generate new object
    (2) call init with only 'name' to load object for further processing
'''

from os.path import exists, join
import numpy as np
import pandas as pd
from copy import deepcopy
import re
import glob
from scipy import signal
import matplotlib.pyplot as plt
from spikeinterface.extractors.neoextractors import SpikeGLXRecordingExtractor
from spectral_connectivity import Multitaper, Connectivity
from scipy.signal import filtfilt, remez
from tqdm import tnrange
from tqdm.notebook import tqdm as tdqm
from matplotlib import gridspec

class LFP:

    '''LFP and related functions from a single session

    Attributes:
        name (string):               path to LFP file
        rec_file_path (string):      path to spikeGLX folder containing lf.bin
        lfp (ts x channels float):   array of LFP voltage values
        timestamps (1xts float):     array of timestamps
        start (int):                 start time (s)
        end (int):                   end time (s)
        rate (int):                  samples/s of downsampled stream
        n_chan (int):                # channels
        channels (list of ints):     indices of channels
        downsample_factor (int):     downsample factor
        max_theta_channel (int):     index of channel with highest theta power
                                        within user-specified channel subset & time subset
    Methods:
        load
        save
        subset_by_channel
        subset_by_time
        calc_max_theta_channel

    Functions:
        get_inclusive_indices
        multitaper_filtered_power
        plot_spectrogram
        plot_psd
        design_filter
        bandpass_filter
        hilbert_envelope_phase_freq
        get_cycle_intervals
    '''

    def __init__(self, name="", rec_file_path="", start=None, end=None, n_chan=384, 
                 probe_type=1, imec=0, downsample_factor=4, channels=None):
        '''Loads or builds spiketimes and relevant variables

        Parameters:
            name (string):              optional, path to LFP file
            rec_file_path (string):     optional, path to spikeGLX folder with lf.bin
            start (int):                optional, start time (sec)
            end (int):                  optional, end time (sec)
            n_chan (int):               optional, # channels
            probe_type (int):           optional, NPX probe type (1 or 2)
            downsample_factor (int):    optional, downsample factor
            channels (list of ints):    optional, indices of channels to include
        '''

        self.name = name

        # if instance already defined
        #if exists(self.name):
            #if ('synced' in self.name):
                #self.load(synced = True)
            #else: 
                #self.load(synced = False)
    
        if exists(self.name):
            self.loadsyncedfile()
            
        else:
            if name:
                raise Exception(f"Cannot find file {self.name}")
            self.rec_file_path = rec_file_path
            self.start = start
            self.end = end
            self.downsample_factor = downsample_factor
            self.rate = 2500
            if channels is None:
                channels = list(range(n_chan))
            self.n_chan = len(channels)
            self.channels = channels

            # design filter
            hardware_filter = signal.butter(
                1, Wn=[0.5, 500], btype='band', fs=self.rate)
            # preprocess_filter = signal.butter(3, Wn=[1,500], btype='band', fs=self.rate) #catGT specs
            preprocess_filter = signal.butter(
                2, Wn=[0.1, 300], btype='band', fs=self.rate)  # Frank lab specs

            lfp_extractor = SpikeGLXRecordingExtractor(self.rec_file_path, \
                                                       stream_id=f'imec{imec}.lf')
                                                       
            if self.start is None:
                self.start = 0
            if self.end is None:
                ap_meta_file = glob.glob(join(rec_file_path,f'*.imec{imec}.ap.meta'))
                file_stream = open(ap_meta_file[0])
                for line in file_stream:
                    if 'fileTimeSecs' in line:
                        self.end = float(re.findall(r'(\d+\.\d+)', line)[0])
                                                       
            # ts x channels empty ndarray
            n_samp = int((self.end*self.rate-self.start * \
                         self.rate)/self.downsample_factor)+1
            
            self.lfp = np.empty((n_samp, self.n_chan))

            #print('beginning filtering each channel to correct for phase shift & impose filter')
            for i, c in enumerate(channels):
                # subsetted by channel to reduce RAM requirements
                lfp_c = lfp_extractor.get_traces(channel_ids=[f"imec{imec}.lf#LF{c}"],
                                                start_frame=int(self.start*self.rate),
                                                end_frame=int(self.end*self.rate))
                lfp_c = lfp_c.T

                if probe_type==1:
                    # correct for analog filter phase shift
                    # reverse, filter in 1 direction, and reverse again
                    lfp_c = np.flip(signal.lfilter(*hardware_filter, np.flip(lfp_c)))
    
                    # initial filtering (similar to what catGT would have done)
                    lfp_c = signal.filtfilt(*preprocess_filter, lfp_c)

                # downsample to reduce filesize
                downsampled_lfp = signal.decimate(
                    lfp_c, self.downsample_factor)
                # the sglx recording extractor sometimes cuts off the last few LFP datapoints
                # so leaves those blank in the LFP array
                self.lfp[0:len(downsampled_lfp[0]), i] = downsampled_lfp

            # store timestamps for future temporal filtering
            self.rate = self.rate/self.downsample_factor  # new downsampled rate
            self.timestamps = np.arange(0, self.end-self.start, 1/self.rate)

    def load(self, synced = True):
        '''Load file'''
        # load metadata as text file, data as csv
        name = self.name

        if synced == True:
            if name[-5] == str(2):
                meta_name = name[:-11] +'_metadata.txt'
            else:
                meta_name = name[:-10]+'_metadata.txt'
        else:
            meta_name = name[:-4]+'_metadata.txt'
        file = open(meta_name, 'r')
        self.__dict__ = eval(file.read())
        self.name = name
        
        with open(self.name, "r") as file:
            lfp_df = pd.read_csv(file)
        self.timestamps = np.array(lfp_df['Timestamps'])
        self.lfp = np.array(lfp_df)
        self.lfp = self.lfp[:, 1:]  # remove Timestamps column

    def loadsyncedfile(self):
        '''Load file'''    
        file = self.name
        lfp_df = pd.read_csv(file)
        self.timestamps = np.array(lfp_df['Timestamps'])
        self.lfp = np.array(lfp_df)
        self.lfp = self.lfp[:, 1:]  # remove Timestamps column
        self.rate = 625 #supply sampling rate metadata

    def save(self):
        '''Save to file'''
        # save metadata as text file, data as csv
        lfp = deepcopy(self.lfp)
        self.channels = self.channels[0]
        del self.lfp
        timestamps = deepcopy(self.timestamps)
        del self.timestamps

        meta_name = self.name[:-4]+'_metadata.txt'
        with open(meta_name, 'w') as file:
            file.write(str(self.__dict__))

        ts_df = pd.DataFrame(data=timestamps, columns=['Timestamps'])
        lfp_df = pd.DataFrame(data=lfp)
        lfp_df = pd.concat([ts_df, lfp_df], axis=1)
        csv_name = self.name[:-4]+'.csv'
        lfp_df.to_csv(csv_name, index=False)

        self.lfp = lfp
        self.timestamps = timestamps

    # %% Subsetting functions
    def subset_by_channel(self, subset_channels):
        '''Subset lfp on specific channels, eg after running electrodes.subset_by_location()'''
        self.lfp = self.lfp[:, subset_channels]
        try:
            self.n_chan = len(subset_channels)
        except:
            self.n_chan = 1
            

    def subset_by_time(self, intervals):
        '''Subset lfp within windows, e.g. during epochs or running'''
        if intervals.ndim == 1:  # if just 1 interval
            # convert vector to 2D array with 1 row
            intervals = intervals[np.newaxis]

        # append all timestamps within the windows (inclusive)
        subset_timestamps = np.empty(0)
        subset_indices = np.empty(0)
        for i in intervals:
            subset_timestamps = np.append(subset_timestamps,
                                          self.timestamps[(self.timestamps >= i[0])
                                                          & (self.timestamps <= i[1])])
            subset_indices = np.append(subset_indices,
                                       np.asarray(np.where((self.timestamps >= i[0])
                                                           & (self.timestamps <= i[1]))))

        # append all lfp samples within the windows (inclusive)
        subset_indices = subset_indices.astype(int)
        subset_lfp = self.lfp[subset_indices, :]

        self.lfp = subset_lfp
        self.timestamps = subset_timestamps

    # %% Calculating features
    #def calc_max_theta_channel(self, channels, interval):
        #'''Calculate & store the channel with the highest theta power'''
        #lfp_theta = deepcopy(self)
        #lfp_theta.subset_by_time(interval)
        #lfp_theta.subset_by_channel(channels)
        #power = multitaper_filtered_power(
            #lfp_theta.lfp, low=THETA[0], high=THETA[1])
        #self.max_theta_channel = channels[np.argmax(power)]

    def calc_max_theta_channel(self, channels, interval):
        '''Calculate & store the channel with the highest theta power'''
        lfp_theta = deepcopy(self)
        lfp_theta.subset_by_time(interval)
        power = multitaper_filtered_power(
            lfp_theta.lfp, low=THETA[0], high=THETA[1])
        self.max_theta_channel = channels[np.argmax(power)]

# %% Filter parameters
# freq band = [low, high, bandpass]
#THETA = [5, 11, 1] Emily's value from Frank Lab
THETA = [6, 12, 1] 
SLOW_WAVE = [1, 4, 0.5]
SPINDLE = [12, 18, 1]
SHARP_WAVE = [0, 30, 5]
SLOW_GAMMA = [20, 50, 5]
FAST_GAMMA = [50, 110, 5]
RIPPLE = [125, 200, 5]

# %% External functions

def plot_power_vs_depth(lfp, rate = 625, low = 6, high = 12, window = 0.5):
    '''Plots & saves theta power, # clusters, & atlas registration vs channels'''
    #get power
    multi = Multitaper(lfp, sampling_frequency=rate,
                   time_window_duration=window, time_window_step=window)
    c = Connectivity.from_multitaper(multi)  # shape: windows x freqs x channels

    # find the target bands (inclusive)
    low_idx, high_idx = get_inclusive_indices(low, high, c.frequencies)

    # extract power in the theta band and timestamps
    power = np.mean(np.mean(c.power()[:, low_idx:high_idx, 0].squeeze(), 1),0)

    #plot theta power for each channel & number of clusters at each depth
    #with overlay of atlas registration
    fig, ax = plt.subplots()
    ax.plot(power, c='black', label='Theta')
    ax.set(xlabel='Channel', ylabel='Power')

def get_inclusive_indices(low, high, freqs):
    '''Find the indices of the target band (inclusive)
    Can also use np.argmin(np.abs(c.frequencies - freqs[0]))
    But this method ensures freqs[low_idx]<=low and freqs[high_idx]>=high'''
    low_idx = 0
    high_idx = len(freqs)-1
    for i, freq in enumerate(freqs):
        if (freq > low) and (freqs[i-1] <= low):
            low_idx = i-1
        if (freq >= high) and (freqs[i-1] < high):
            high_idx = i
            break
    return low_idx, high_idx


def multitaper_filtered_power(lfp, rate=625, low=1, high=300, window=0.5):
    '''Use multi-taper spectrogram to calculate power in freq. window'''
    # calculate spectrogram
    m = Multitaper(lfp, sampling_frequency=rate,
                   time_window_duration=window, time_window_step=window)
    c = Connectivity.from_multitaper(m)  # shape: windows x freqs x channels

    # get power in freq band
    band_start, band_end = get_inclusive_indices(low, high, c.frequencies)
    power = np.mean(c.power()[:, band_start:band_end, :].squeeze(), 1)
    total_power = np.mean(power, 0)
    return power, total_power


def plot_spectrogram(lfp, rate=625, low=1, high=300, low_display=1,
                     high_display=300, window=0.5, save = False, save_file = None):
    '''Use multi-taper spectrogram to plot power across all freqs in window on a single channel'''
    # calculate spectrogram
    m = Multitaper(lfp, sampling_frequency=rate,
                   time_window_duration=window, time_window_step=window)
    c = Connectivity.from_multitaper(m)  # shape: windows x freqs x channels

    # find the target bands (inclusive)
    low_idx, high_idx = get_inclusive_indices(low, high, c.frequencies)
    low_disp_idx, high_disp_idx = get_inclusive_indices(
        low_display, high_display, c.frequencies)

    # extract power and timestamps
    power = np.mean(c.power()[:, low_idx:high_idx, 0].squeeze(), 1)
    n_samples = int(window * rate)
    timestamps = np.arange(0, lfp.shape[0])
    index = timestamps[np.arange(1, power.shape[0] * n_samples + 1, n_samples)]
    power = pd.DataFrame(power, index=index)
    timestamps = power.index/rate

    # plot spectrogram
    gs = gridspec.GridSpec(20, 1, hspace=1.5)
    f = plt.figure(figsize=(10, 13))
        
    # spectrogram
    ax0 = plt.subplot(gs[4:20])
    ax0.pcolormesh(timestamps, c.frequencies[low_disp_idx:high_disp_idx],
                  c.power()[:, low_disp_idx:high_disp_idx, 0].squeeze().T, cmap='viridis')
    ax0.set(xlabel='Time (s)', ylabel='Frequency (Hz)')

    # plot theta power over time
    ax1 = plt.subplot(gs[:4])
    ax1.plot(timestamps, power.values)
    ax1.set(xlabel='Time (s)', ylabel='Theta Power (uV)',
           xlim=[timestamps[0], timestamps[-1]])
    
    if (save == True) and (save_file is not None):
        plt.savefig(save_file)
    plt.show()

def plot_psd(lfp, rate=625, save = False, save_file = None):
    '''Plots PSD for a single channel'''
    freqs, psd = signal.welch(np.squeeze(lfp), rate)
    plt.semilogy(freqs, psd)
    plt.xlim([0, 200])
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('PSD (V**2/Hz)')
    if (save == True) and (save_file is not None):
        plt.savefig(save_file)
    plt.show()

# following 2 functions adapted from
# https://github.com/Eden-Kramer-Lab/ripple_detection/blob/master/ripple_detection/core.py


def design_filter(rate=625, low=1, high=300, pass_band=1):
    '''Returns a remez (aka equiripple) bandpass filter
    Parameters
    ----------
    samp_rate : sampling frequency (Hz)
    freqs : tuple of low & high freqs (Hz)
    pass_band : width of pass band (Hz)
    Returns
    -------
    filtered_data : array_like, shape (n_time,)
    '''
    order = 101
    nyquist = 0.5*rate
    desired = [0, low - pass_band, low,
               high, high + pass_band, nyquist]
    return remez(order, desired, [0, 1, 0], Hz=rate), 1.0


def bandpass_filter(lfp, rate=625, low=1, high=300, pass_band=1):
    '''Returns a bandpass filtered signal
    Parameters
    ----------
    data : array_like, shape (n_time,)
    samp_rate : sampling frequency (Hz)
    freqs : tuple of low & high freqs (Hz)
    pass_band : width of pass band (Hz)
    Returns
    -------
    filtered_data : array_like, shape (n_time,)
    '''
    if (lfp.ndim>1) and (lfp.shape[1]>1):
        raise Exception('Can only filter 1 channel at a time')
    filt_num, filt_denom = design_filter(rate, low, high, pass_band)
    is_nan = np.isnan(lfp)
    filtered_data = np.full_like(lfp, np.nan)
    filtered_data[~is_nan] = filtfilt(
        filt_num, filt_denom, lfp[~is_nan], axis=0)
    return filtered_data


def hilbert_envelope_phase_freq(lfp, win=15, rate=625):
    '''Hilbert transform to get envelope, phase, and frequency of LFP'''
    # smooth filtered signal
    filter_coef = signal.get_window('hann', win, fftbins=False)
    filter_coef /= filter_coef.sum()
    # filtfilt is expecting (channel x timestamps) shape
    smoothed_lfp = signal.filtfilt(filter_coef, 1, lfp.T)
    # multiple by -1 so cycle is peak-to-peak instead of trough-to-trough
    smoothed_lfp = smoothed_lfp.T*-1 
    
    # hilbert transform
    analytic_signal = signal.hilbert(smoothed_lfp, axis=0)
    envelope = np.abs(analytic_signal)
    inst_phase = np.unwrap(np.angle(analytic_signal))
    #inst_freq = (np.diff(inst_phase)/(2.0*np.pi)*rate)
    inst_freq = (np.diff(inst_phase, axis=0)/(2.0*np.pi)*rate)
    return envelope, inst_phase, inst_freq


def get_cycle_intervals(lfp, timestamps, win=15, rate=625):
    '''Given a (theta-filtered) 1-channel LFP trace, returns a (i,2) nparray of i intervals
    where [intervals[i,0], intervals[i,1]] spans a single (theta) cycle
    '''
    _, phase, _ = hilbert_envelope_phase_freq(lfp, win, rate)
    new_cycle = np.diff(phase, axis=0) < 0
    new_cycle = np.append(new_cycle, [False])  # match length to timestamps
    curr_start = timestamps[0]
    intervals = []
    start_indices = [0]
    for t, time in enumerate(timestamps):
        # when phase shifts from high (~2pi) to low (~0), start a new cycle
        if new_cycle[t]:
            intervals.append([curr_start, timestamps[t-1]])
            start_indices.append(t-1)
            curr_start = timestamps[t-1]
        # end of the array without hitting an interval-ending new cycle
        elif t == len(timestamps)-1:
            intervals.append([curr_start, timestamps[t]])
            start_indices.append(t)

    start_indices = start_indices[:-1]  # remove end index
    return intervals, start_indices
