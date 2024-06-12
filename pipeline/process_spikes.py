import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import sem
import scipy.io
from tqdm import trange
import itertools 

def draw_raster(cell_IDs, cell_ID, ax, posx, trials, Y, color='k', minimalist=False):
    '''
    Make raster plot of spikes by position and trial.
    Can either input a single cell ID and one subplot axis,
    or a series of cell IDs and the corresponding 1D axis object.
    
    Params
    ------
    cell_IDs : tuple
        to index into the spike matrix
    cell_ID: tuple
        cell you'd like to plot spikes for
    ax : subplot axes object
        single subplot axis for one raster or multiple axes for many rasters
    posx : ndarray
        position at each observation; shape (n_obs)
    trial : ndarray
        trial number at each observation; shape (n_obs)
    Y : ndarray
        matrix of spike occurances; shape (n_obs, n_cells)
    color : string or 3-element list
        color to plot raster; optional, default is 'k'
    minimalist : bool
        exclude axis labels and spike count from title; optional, default is False
    '''
    c = cell_ID
    sdx = Y[:, np.where(cell_IDs==c)[0][0]].astype(bool)
    ax.plot(posx[sdx], trials[sdx], '.', c='k', markersize=2, alpha=.3)
    if minimalist:
        ax.set_title('cell ' + str(c))
        ax.tick_params(labelleft=False, labelbottom=False)
    else:
        ax.set_title('cell ' + str(c) + ', ' + str(int(np.sum(Y[:, np.where(cell_IDs==c)[0][0]]))) + ' spikes')
        #ax.set_xlabel('Virtual Track Position (cm)')
        ax.set_ylabel('Trials (n)')
    ax.set_xlim((0, 400))
    ax.set_ylim((0, np.max(trials)))
        
def ogdraw_raster(cell_IDs, m, s, save_folder, ax, posx, trials, Y, color='k', minimalist=False, save = False):
    '''
    Make & save composite raster plot of spikes by position and trial.
    Can either input a single cell ID and one subplot axis,
    or a series of cell IDs and the corresponding 1D axis object.
    
    Params
    ------
    cell_IDs : tuple
        to index into the spike matrix
    m : str
        mouse name 
    s : str
        session name
    save_folder : str
        where to save composite image
    ax : subplot axes object
        single subplot axis for one raster or multiple axes for many rasters
    posx : ndarray
        position at each observation; shape (n_obs)
    trial : ndarray
        trial number at each observation; shape (n_obs)
    Y : ndarray
        matrix of spike occurances; shape (n_obs, n_cells)
    color : string or 3-element list
        color to plot raster; optional, default is 'k'
    minimalist : bool
        exclude axis labels and spike count from title; optional, default is False
    save : bool
        determines whether to save figure or not; optional, default is False
    '''
    fig = plt.figure(figsize = (24,12))
    fig.suptitle('Mouse ' + m + ' Session ' + s)
    nrow = 5
    ncol = int(np.round(len(cell_IDs)/nrow) + 1)
    
    if len(cell_IDs) > 1:
        for i, c in enumerate(cell_IDs):
            sdx = Y[:, np.where(cell_IDs==c)[0][0]].astype(bool)
            idx = np.where(cell_IDs==c)[0][0] + 1 
            ax = fig.add_subplot(nrow,ncol,idx)
            ax.plot(posx[sdx], trials[sdx], '.', c=color, markersize=0.3, alpha=.3)
            if minimalist:
                ax.set_title('cell ' + str(c), fontsize = 8)
                ax.tick_params(labelleft=False, labelbottom=False)
            else:
                ax.set_title('cell ' + str(c) + ', ' + str(int(np.sum(Y[:, np.where(cell_IDs==c)[0][0]]))) + ' spikes')
                ax.set_xlabel('Virtual Track Position (cm)')
                ax.set_ylabel('Trials (n)')
            ax.set_xlim((0, 400))
            ax.set_ylim((0, np.max(trials)))
    else:
        c = cell_IDs
        sdx = Y[:, np.where(cell_IDs==c)[0][0]].astype(bool)
        ax.plot(posx[sdx], trials[sdx], '.', c=color, markersize=0.3, alpha=.3)
        if minimalist:
            ax.set_title('cell ' + str(c))
            ax.tick_params(labelleft=False, labelbottom=False)
        else:
            ax.set_title('cell ' + str(c) + ', ' + str(int(np.sum(Y[:, np.where(cell_IDs==c)[0][0]]))) + ' spikes')
            ax.set_xlabel('Virtual Track Position (cm)')
            ax.set_ylabel('Trials (n)')
        ax.set_xlim((0, 400))
        ax.set_ylim((0, np.max(trials)))
        
    plt.subplots_adjust(hspace = 0.2, wspace = 0.2)
    
    if save:
        fig.savefig(save_folder + m + '_' + s + '_compositeraster.png', dpi = 100)

def draw_rasters(cell_IDs, ax, posx, trials, Y, color='k', minimalist=False):
    '''
    Make raster plots of spikes by position and trial given a series of cell IDs.
    
    Params
    ------
    cell_IDs : tuple
        to index into the spike matrix
    ax : subplot axes object
        single subplot axis for one raster or multiple axes for many rasters
    posx : ndarray
        position at each observation; shape (n_obs)
    trial : ndarray
        trial number at each observation; shape (n_obs)
    Y : ndarray
        matrix of spike occurances; shape (n_obs, n_cells)
    color : string or 3-element list
        color to plot raster; optional, default is 'k'
    minimalist : bool
        exclude axis labels and spike count from title; optional, default is False
    '''
    if len(cell_IDs) > 1:   
        nrow = 6
        ncol = int(np.round(len(cell_IDs)/nrow) + 1)
        fig, ax = plt.subplots(len(cell_IDs),1,squeeze=False)
        fig.subplots_adjust(hspace=8, wspace=8)
        
        plot = 0
        for i, c in enumerate(cell_IDs):
            sdx = Y[:, np.where(cell_IDs==c)[0][0]].astype(bool)
            ax[plot,0].plot(posx[sdx], trials[sdx], '.', c=color, markersize=6, alpha=.3)
            if minimalist:
                ax[plot,0].set_title('cell ' + str(c))
                ax[plot,0].tick_params(labelleft=False, labelbottom=False)
            else:
                ax[plot,0].set_title('cell ' + str(c) + ', ' + str(int(np.sum(Y[:, np.where(cell_IDs==c)[0][0]]))) + 'spikes')
                ax[plot,0].set_xlabel('Virtual Track Position (cm)')
                ax[plot,0].set_ylabel('Trials (n)')
            ax[plot,0].set_xlim((0, 400))
            ax[plot,0].set_ylim((0, np.max(trials))) 
            plot = plot + 1
            
        
    else:
        print('oops there is only one cell here! use draw_raster instead!')

def tuning_curve(x, Y, dt, b, smooth=True, l=2, SEM=False, occupancy=True):
    '''
    Params
    ------
    x : ndarray
        variable of interest by observation; shape (n_obs, )
    Y : ndarray
        spikes per observation; shape (n_obs, n_cells)
    dt : int
        time per observation in seconds
    b : int
        bin size
    smooth : bool
        apply gaussian filter to firing rate; optional, default is True
    l : int
        smoothness param for gaussian filter; optional, default is 2
    SEM : bool
        return SEM for FR; optional, default is False
    occupancy : bool
        return occupancy (dwell time in each bin); optional, default is True

    Returns
    -------
    firing_rate : ndarray
        trial-averaged, binned firing rate for each cell; shape (n_bins, n_cells)
    centers : ndarray
        center of each bin
    occ : ndarray
        dwell time in each bin; shape (n_bins, n_cells)
    '''

    edges = np.arange(0, np.max(x) + b, b)
    centers = (edges[:-1] + edges[1:])/2
    b_idx = np.digitize(x, edges)
    if np.max(x) == edges[-1]:
        b_idx[b_idx==np.max(b_idx)] = np.max(b_idx) - 1
    unique_bdx = np.unique(b_idx)
    
    # find FR in each bin
    firing_rate = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    spike_sem = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    occ = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    for i in range(unique_bdx.shape[0]):
        spike_ct = np.sum(Y[b_idx == unique_bdx[i], :], axis=0)
        occupancy = dt * np.sum(b_idx==unique_bdx[i])
        occ[i, :] = occupancy
        spike_sem[i, :] = sem(Y[b_idx == unique_bdx[i], :]/dt, axis=0)
        firing_rate[i, :] = spike_ct / occupancy
    
    if smooth:
        firing_rate = gaussian_filter1d(firing_rate, l, axis=0, mode='wrap')
        spike_sem = gaussian_filter1d(spike_sem, l, axis=0, mode='wrap')
        occ = gaussian_filter1d(occ, l, axis=0, mode='wrap')
    
    if SEM:
        if occupancy:
            return firing_rate, centers, spike_sem, occ
        else:
            return firing_rate, centers, spike_sem
    else:
        if occupancy:
            return firing_rate, centers, occ
        else:
            return firing_rate, centers

# find FR by trial
def tuning_curve_bytrial(x, trial, Y, dt, b, smooth=True, normalize=False, occupancy=True):
    '''
    Params
    ------
    x : ndarray
        variable of interest by observation; shape (n_obs, )
    trial : ndarray
        trial num for each observation; shape (n_obs, )
    Y : ndarray
        spikes per observation; shape (n_obs, n_cells)
    dt : int
        time per observation in seconds
    b : int
        bin size
    smooth : bool
        apply gaussian filter to firing rate; optional, default is True
    normalize : bool
        normalize the firing rate of each cell such that its max FR is 1, min is 0;
        optional, default is False
    occupancy : bool
        return occupancy (dwell time in each bin); optional, default is True

    Returns
    -------
    firing_rate : ndarray
        binned firing rate for each trial for each cell; shape (n_trials, n_bins, n_cells)
    centers : ndarray
        center of each bin
    occ : ndarray
       dwell time in each bin; shape (n_bins, n_cells)
    '''
    edges = np.arange(0, np.max(x) + b, b)
    centers = (edges[:-1] + edges[1:])/2
    b_idx = np.digitize(x, edges)
    if np.max(x) == edges[-1]:
        b_idx[b_idx==np.max(b_idx)] = np.max(b_idx) - 1
    unique_bdx = np.unique(b_idx)

    # find FR in each bin
    firing_rate = np.zeros((np.unique(trial).shape[0], unique_bdx.shape[0], Y.shape[1]))
    occ = np.zeros((np.unique(trial).shape[0], unique_bdx.shape[0], Y.shape[1]))
    for j in range(unique_bdx.shape[0]):
        idx1 = (b_idx == unique_bdx[j])
        for i, t in enumerate(np.unique(trial)):
            idx = idx1 & (trial == t)
            if np.sum(idx)==0:
                #print('warning: zero occupancy!')
                firing_rate[i, j, :] = firing_rate[i, j-1, :]
                occ[i, j, :] = 0
            else:    
                spike_ct = np.sum(Y[idx, :], axis=0)
                occupancy = dt * np.sum(idx)
                occ[i, j, :] = occupancy
                firing_rate[i, j, :] = spike_ct / occupancy
    if smooth:
        firing_rate = gaussian_filter1d(firing_rate, 2, axis=1, mode='wrap')

    if normalize:
        for c in range(firing_rate.shape[2]):
            firing_rate[:, :, c] = (firing_rate[:, :, c] - np.min(firing_rate[:, :, c]))/np.max(firing_rate[:, :, c] - np.min(firing_rate[:, :, c]))
    
    if occupancy:
        return firing_rate, centers, occ
    else: 
        return firing_rate, centers
    
# find FR by trial
def tuning_smoothedcurve_bytrial(x, trial, Y, dt, sigma, b, smooth=True, normalize=False, occupancy=True):
    '''
    Params
    ------
    x : ndarray
        variable of interest by observation; shape (n_obs, )
    trial : ndarray
        trial num for each observation; shape (n_obs, )
    Y : ndarray
        spikes per observation; shape (n_obs, n_cells)
    dt : int
        time per observation in seconds
    sigma : int    
        smoothing factor    
    b : int
        bin size
    smooth : bool
        apply gaussian filter to firing rate; optional, default is True
    normalize : bool
        normalize the firing rate of each cell such that its max FR is 1, min is 0;
        optional, default is False
    occupancy : bool
        return occupancy (dwell time in each bin); optional, default is True

    Returns
    -------
    firing_rate : ndarray
        binned firing rate for each trial for each cell; shape (n_trials, n_bins, n_cells)
    centers : ndarray
        center of each bin
    occ : ndarray
       dwell time in each bin; shape (n_bins, n_cells)
    '''
    edges = np.arange(0, np.max(x) + b, b)
    centers = (edges[:-1] + edges[1:])/2
    b_idx = np.digitize(x, edges)
    if np.max(x) == edges[-1]:
        b_idx[b_idx==np.max(b_idx)] = np.max(b_idx) - 1
    unique_bdx = np.unique(b_idx)

    # find FR in each bin
    firing_rate = np.zeros((np.unique(trial).shape[0], unique_bdx.shape[0], Y.shape[1]))
    occ = np.zeros((np.unique(trial).shape[0], unique_bdx.shape[0], Y.shape[1]))
    for j in range(unique_bdx.shape[0]):
        idx1 = (b_idx == unique_bdx[j])
        for i, t in enumerate(np.unique(trial)):
            idx = idx1 & (trial == t)
            if np.sum(idx)==0:
                #print('warning: zero occupancy!')
                firing_rate[i, j, :] = firing_rate[i, j-1, :]
                occ[i, j, :] = 0
            else:    
                spike_ct = np.sum(Y[idx, :], axis=0)
                occupancy = dt * np.sum(idx)
                occ[i, j, :] = occupancy
                firing_rate[i, j, :] = spike_ct / occupancy
    if smooth:
        firing_rate = gaussian_filter1d(firing_rate, sigma, axis=1, mode='wrap')

    if normalize:
        for c in range(firing_rate.shape[2]):
            firing_rate[:, :, c] = (firing_rate[:, :, c] - np.min(firing_rate[:, :, c]))/np.max(firing_rate[:, :, c] - np.min(firing_rate[:, :, c]))
    
    if occupancy:
        return firing_rate, centers, occ
    else: 
        return firing_rate, centers

#calculate stability 
def stability(FR_1, FR_2):
    '''
    Calculate stability between two tuning curves

    Params
    ------
    FR_1 : ndarray
        trial-averaged firing rate for each cell; shape (n_bins, n_cells)
    FR_2 : ndarray
        same as FR_1; match cells

    Returns
    -------
    stability_scores : ndarray
        correlation of each cell's tuning curve across FR_1 and 2; shape (n_cells,)
    '''
    # check inputs
    if FR_1.shape != FR_2.shape:
        raise ValueError("input shapes do not match!")

    # mean center and normalize
    FR_1 -= np.mean(FR_1, axis=0)
    FR_1 /= np.linalg.norm(FR_1, axis=0) + 1e-6
    FR_2 -= np.mean(FR_2, axis=0)
    FR_2 /= np.linalg.norm(FR_2, axis=0) + 1e-6
    
    # calculate correlation
    stability_scores = np.diag(np.dot(FR_1.T, FR_2))

    return stability_scores

def cross_trial_correlation(spatial_map, shift=False, max_shift=5):
    '''
    Find correlation of spatial maps across trials for a given cell, option to allow for a positional shift (drift)
    
    Params
    ------
    spatial_map : ndarray
        firing rate by position and trial for a single cell; shape (n_trials, n_pos_bins)
    shift : bool
        if True, calculates best correlation across pairs of trials allowing for shifts in position up to max_shift;
        optional, default is False
    max_shift : int
        maximum number of position bins map can shift by, default is 5
        
    Output
    ------
    similarity : ndarray
        similarity of each trial to all others; shape (n_trials, n_trials)
    shifts : ndarray
        if shift=True, shift needed to find maximum similarity; shape (n_trials, n_trials)
    '''    
    num_trials = spatial_map.shape[0]
        
    # normalize and mean center
    spatial_map -= spatial_map.mean(axis=1, keepdims=True)
    spatial_map /= np.linalg.norm(spatial_map, axis=1, keepdims=True) + 1e-6
    
    if shift:
        # initialize matrix to store shifts
        shifts = np.zeros((num_trials, num_trials), dtype='int')
        
        # set shift
        start = (spatial_map.shape[1] // 2) - max_shift
        stop = (spatial_map.shape[1] // 2) + max_shift
    else:
        shifts = None

    # initialize matrix to store similarity values
    similarity = np.zeros((num_trials, num_trials))
    
    # calculate maximum similarity for each pair of trials
    if shift:
        for i, j in itertools.combinations(range(num_trials), 2):
            tmp = np.correlate(spatial_map[i], spatial_map[j], 'same')[start:stop+1]
            shifts[i, j] = np.argmax(tmp) - max_shift
            shifts[j, i] = shifts[i, j]
            similarity[i, j] = tmp[shifts[i, j]]
    else:
        similarity = np.dot(spatial_map, spatial_map.T)
        
    return similarity, shifts

def get_coordinates(tip, entry, distances):
    '''
    Convert distance from tip to 3D brain coordinates (ML, AP, DV)

    Params:
    ------
    tip : ndarray
        coordinates of probe tip; shape(3)
    entry : ndarray
        coordinates of probe entry into MEC; shape(3)
    distances : ndarray
        distance of each cell from probe tip; shape(n_cells)
        
    Returns:
    -------
    coords : ndarray
        coordinates for each cell; shape(n_cells, 3)
    '''
    probe_vec = (entry - tip) # vector connecting tip coords to entry coords
    l = np.linalg.norm(probe_vec) # length of the probe
    coords = tip[None, :] + (distances[:, None]/l) * probe_vec[None, :]
    
    return(coords)


def in_MEC(coords, slices, ML_idx):
    '''
    takes a list of cell coordinates and compares to a list of reference coordinates
    determines if each cell is in MEC

    Params:
    ------
    coords : ndarray
        list of 3D coordinates; shape(n_cells, 3)
    slices : list
        contains slice quadrants for each sagittal slice; size(n_slices)
    slice_quads : list
        coordinates [min_DV, max_DV, max_AP] defining MEC quadrants for a given slice
        multiple rectangles are given when needed to better bound MEC; size(n_quadrants)
        (min AP is always 0)
    ML_idx : ndarray
        set of ML values corresponding to each slice; shape(n_slices)
        
    a given cell is in MEC if it satisfies:
        (min_DV <= cell_DV <= max_DV) & (cell_AP <= max_AP)
        for the slice nearest to cell_ML

    output is a bool of length(n_cells), True if cell is in MEC
    '''
    MEC_idx = np.ones(coords.shape[0])
    
    for i, c in enumerate(coords):
        # get slice idx
        if c[0] > -225:
            idx = np.argmin(np.abs(ML_idx - c[0]))
            slice_quads = slices[idx]
        else: # too far medial
            MEC_idx[i] = 0
            continue
        
        # check quadrants in that slice
        for j, q in enumerate(slice_quads):
            if (q[0] <= c[-1] <= q[1]) & (c[1] <= q[-1]):
                continue
            elif j == len(slice_quads):
                MEC_idx[i] = 0
    
    return MEC_idx.astype(bool)