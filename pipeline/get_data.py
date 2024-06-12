""" Scripts for Loading and Formating Data """
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d
import scipy.io
import h5py
import hdf5storage
from tqdm import trange

def loadData(path_to_data):
    """Loads the neuropixels matlab data struct and extracts relevant variables

    Returns
    -------
    data : dict
        dict containing behavioral and spiking data
        data['sp'] gives dict of spiking data

    """
    # load data
    d = loadmat_sbx(path_to_data)

    #if 'data' in d: # struct is from silicon probe data
        #data = d['data']
        #print(data.keys())
   # else:
       # print('could not recognize data format!')

    return d

def loadBigData(path_to_data):
    """Loads v7.3 matlab data structs

    Returns
    -------
    d : dict

    """
    # load data
    ca_dat = {}
    with h5py.File(path_to_data, 'r') as f:
        for k, v in f.items():
            ca_dat[k] = v

    return _check_keys(ca_dat)

def formatData(d, tbin=None, get_vid=True):
    """
    Parameters
    ----------
    d : struct
        struct containing behavior and spiking data, obtain with loadData
    tbin : int
        time bin (in seconds) for binning data, optional
    get_vid : bool
        store video params; optional, default is true

    Returns
    -------
    X : ndarray
        behavioral data, binned by tbin; shape (n_bins, n_vars)
    X_labels : list
        variable name associated with each column of X
    Y : ndarray
        spike data from good cells; shape (n_bins, n_goodcells)
    """

    ''' get behavioral params '''
    b_vars = []
    X_labels = []
    posx = d['posx']
    b_vars.append(d['posx'])
    X_labels.append('posx')
    post = d['post']
    b_vars.append(d['post'])
    X_labels.append('post')
    lickx = d['lickx']
    b_vars.append(d['lickx'])
    X_labels.append('lickx')
    lickt = d['lickt']
    b_vars.append(d['lickt'])
    X_labels.append('lickt')    

    dt = np.unique(np.round(np.diff(post),4))
    speed = getSpeed(posx, dt)
    b_vars.append(speed)
    X_labels.append('speed')

    trial = trial_idx(posx)
    b_vars.append(trial)
    X_labels.append('trial')

    ''' store session info if available'''
    if 'session' in d:
        b_vars.append(d['session'])
        X_labels.append('session')

    ''' z-score and normalize pupil and whisk '''
    if get_vid:
        if 'pupil' in d:
            pupil = d['pupil_upsampled']
            pupil = cleanup(pupil, trial)
            b_vars.append(pupil)
            X_labels.append('pupil')

            whisk = d['whisk_upsampled']
            whisk = cleanup(whisk, trial)
            b_vars.append(whisk)
            X_labels.append('whisk')
        else:
            print('no video data found!')

    ''' get spike params '''
    sp = d['sp']  # spike struct
    cluster_id = sp['clu']
    spiket = sp['st']
    templates = sp['spikeTemplates']
    cgs = sp['cgs']  # classification
    cids = sp['cids']  # cell number
    
    ''' filter by speed and position '''
    # build matrix of behavior
    for b in range(len(b_vars)):
        if b == 0:
            A = b_vars[b]
        else:
            A = np.column_stack((A, b_vars[b]))

    # build matrix of spikes based on if cluster was classified good (2) vs. mua (1)
    print(str(np.sum(cgs==2)) + ' good cells out of ' + str(cgs.shape[0])+ ' total')
    cells = cids[cgs == 2]
    d['cells'] = cells
    B = np.zeros((A.shape[0], cells.shape[0]))
    print('spike matrix size pre behavioral filtering = ' + str(B.shape))
    
    for i in trange(cells.shape[0]):
        # get spike times
        st = spiket[cluster_id == cells[i]]
        B[:, i] = spiketrain(post, dt, st)

    # filter by speed and ends of track
    def find(x):
        return x.nonzero()[0]

    speed_to_trash = find(speed < 2)
    pos_to_trash = find((posx < 0) | (posx > 400))
    trash_idx = np.unique(np.concatenate((speed_to_trash, pos_to_trash)))
    keep_idx = np.setdiff1d(np.arange(A.shape[0]), trash_idx)

    A = A[keep_idx, :]
    B = B[keep_idx, :]
    
    '''filter by spikes'''
    cells = d['cells']
    d['cells'] = cells[np.sum(B, axis=0) > 400]
    B = B[:, np.sum(B, axis=0) > 400]
    print('spike matrix size post behavioral filtering = ' + str(B.shape))
    
    ''' bin by time '''
    if tbin == None:
        d['A'] = A
        d['B'] = B
        
        X = A
        Y = B
        
    else:
        post_binned = np.arange(np.min(post), np.max(post) + tbin, tbin)
        tdx = np.digitize(A[:, 3], post_binned)
        unique_tdx = np.unique(tdx)
        X = np.zeros((unique_tdx.shape[0], A.shape[1]))
        Y = np.zeros((unique_tdx.shape[0], B.shape[1]))

        for i in trange(unique_tdx.shape[0]):
            # position - correct for track ends
            positions = A[tdx == unique_tdx[i], 0]
            if any(np.diff(positions) < 0):
                idx = np.zeros(np.diff(positions).shape[0] + 1)
                idx[0] = False
                idx[1:] = np.diff(positions) < 0
                positions[idx.astype(bool)] += 400
            for b in len(b_vars):
                if b == 0: # position
                	X[i, b] = np.mean(positions % 400)
                elif b == 3: # trial
                	continue
                else:
                	X[i, b] = np.mean(A[tdx == unique_tdx[i], b])  # time, speed, etc.
            Y[i, :] = np.sum(B[tdx == unique_tdx[i], :], axis=0)  # spikes
        X[:, 3] = trial_idx(X[:, 0])

    return X, X_labels, Y

""" Helper Functions """
def nan_interp(y):
    def find(x):
        return x.nonzero()[0]
    nans = np.isnan(y)
    y[nans] = np.interp(find(nans), find(~nans), y[~nans])
    return y

def cleanup(a, trial):
    # z-score
    z = a - np.mean(a)
    z /= z.std()
    idx = np.abs(z) > 2
    
    new_a = a.copy()
    new_a[idx] = np.nan
    new_a = nan_interp(new_a)
    
    # normalize within each trial
    for i in range(int(np.max(trial))):
        idx = trial == i
        if idx.sum() == 0:
            continue
        a_min = np.min(new_a[idx])
        a_max = np.max(new_a[idx])
        new_a[idx] = (new_a[idx] - a_min) / (a_max - a_min)
    
    return nan_interp(new_a)


def getSpeed(posx, dt):
    """ Gets instantaneous speed, filters, and smooths

    Parameters
    ---------
    posx : ndarray
        position at each observation point; shape (n_obs,)
    dt : int
        time between each position bin

    Returns
    -------
    speed : ndarray
        speed at each observation point; shape (n_obs,)

    """
    speed = np.zeros(posx.shape[0])
    speed[1:] = np.diff(posx)/dt
    speed[0] = speed[1]

    # throw out extreme values and interpolate
    speed[speed > 150] = np.nan
    speed[speed < -5] = np.nan
    speed = nan_interp(speed)

    # smooth
    import scipy.ndimage.filters as filt
    sigma = 10  # smoothing factor
    speed = filt.gaussian_filter1d(speed,sigma)

    return speed

def rewardtrain(post, timebin, rewardt, index=False):
    """ Finds count in each time bin
    **only works for consistent sampling rate, defined by timebin**

    Parameters
    ----------
    post : ndarray
        time of each observation, in seconds; shape (n_obs,)
    timebin : int
        time-step between each timebin, in seconds
    rewardt : ndarray
        timept for each reward delivery; shape (n_rewards,)
    index : bool
        converts output to bool if it will be used as an index; default is False

    Returns
    -------
    reward_ct : ndarray
        number of rewards per observation; shape (n_obs,)
    """
    reward_ct = np.zeros_like(post)
    reward_ind = np.rint(rewardt / timebin).astype(int)
    creward_ind = reward_ind[np.where(reward_ind <= len(post))] #ignores rewards recorded after the last position observation
    
    idx, cts = np.unique(creward_ind, return_counts=True) #ignores duplicates 
    reward_ct[idx] = cts

    if index:
        reward_ct = reward_ct.astype(bool)
    return reward_ct

def spiketrain(post, timebin, spiket, index=False):
    """ Finds spike count in each time bin
    **only works for consistent sampling rate, defined by timebin**

    Parameters
    ----------
    post : ndarray
        time of each observation, in seconds; shape (n_obs,)
    timebin : int
        time-step between each timebin, in seconds
    spiket : ndarray
        timept for each observed spike; shape (n_spikes,)
    index : bool
        converts output to bool if it will be used as an index; default is False

    Returns
    -------
    spike_ct : ndarray
        number of spikes per observation; shape (n_obs,)
    """
    spike_ct = np.zeros_like(post)
    spike_ind = np.rint(spiket / timebin).astype(int)
    idx, cts = np.unique(spike_ind, return_counts=True)
    spike_ct[idx] = cts

    if index:
        spike_ct = spike_ct.astype(bool)
    return spike_ct

def trial_idx(posx):
    """ get trial number for each observation

    Parameters
    ----------
    posx : ndarray
        position at each observation; shape (n_obs,)

    Returns
    -------
    trial : ndarray
        zero-indexed trial number at each observation; shape (n_obs,)

    """
    trial = np.cumsum(np.diff(posx) < -100)
    trial = np.hstack([0, trial])

    return trial

def find8adjacentElements(test_list):
    ''' 
    Params
    ------
    test_list : ndarray
        1d array to be sorted
    
    Returns
    -------
    neighbors : list
        nested list where each element is a list of 8 adjacent elements to the element with the same 
        index in test_list, adjusting for the first and last four elements and not including 
    '''
    
    neighbors = []
    for idx, ele in enumerate(test_list):
    # Checking for all cases to append
        if idx == 0:
            neighbors.append(test_list[(idx+1):(idx + 9)])
        elif idx == 1:
            neighbors.append(np.concatenate((test_list[(idx - 1)],test_list[(idx+1):(idx + 8)]),axis=None))
        elif idx == 2:
            neighbors.append(np.concatenate((test_list[:idx],test_list[(idx+1):(idx + 7)]),axis=None))
        elif idx == 3:
            neighbors.append(np.concatenate((test_list[:idx],test_list[(idx+1):(idx + 6)]),axis=None))
        elif idx == len(test_list) - 1:
            neighbors.append(np.concatenate((test_list[(idx-8):idx]),axis=None))                     
        elif idx == len(test_list) - 2:
            neighbors.append(np.concatenate((test_list[(idx-7):idx],test_list[(idx + 1):]),axis=None))
        elif idx == len(test_list) - 3:
            neighbors.append(np.concatenate((test_list[(idx-6):idx],test_list[(idx + 1):]),axis=None))
        elif idx == len(test_list) - 4:
            neighbors.append(np.concatenate((test_list[(idx-5):idx],test_list[(idx + 1):]),axis = None))
        else:
            neighbors.append(np.concatenate((test_list[(idx - 4):idx],test_list[(idx+1):(idx + 5)]),axis=None))
    return neighbors 

    return (phi ** np.arange(nt)) * sigma2 / (1 - phi ** 2)

def autocorr(x,lags):
    '''numpy.correlate'''
    mean=x.mean()
    var=np.var(x)
    xp=x-mean
    corr=np.correlate(xp,xp,'full')[len(x)-1:]/var/len(x)

    return corr[:len(lags)]

def find(x):
    return x.nonzero()[0]

def pf(x, alpha, beta):
    # defines psychometric function based on a sigmoid curve with scale beta and center alpha
    return 1. / (1 + np.exp( -(x-alpha)/beta ))

def count_consec(lst):
    # for a list, returns an array counting the number of consecutive times each value has appeared
    consec = [1]
    for x, y in zip(lst, lst[1:]):
        if x == y - 1:
            consec[-1] += 1
        else:
            consec.append(1)
    return consec

def find_skipped(trial, max_trial_num): 
    
    if (len(np.unique(trial)) < max_trial_num): 
        alltrial = np.arange(0, np.max(np.unique(trial)) + 1 , 1)
    else:    
        alltrial = np.arange(0,max_trial_num,1)
    
    skippedtrials = np.setdiff1d(alltrial, np.unique(trial)).astype(int)
    
    return skippedtrials

def contextsortedtrials(alltrial, context, alt_idx = 140):

    #sorts alternation trials based on context info
    contextsort = context.copy()
    contexttail = context[alt_idx:]
    contexttail = 1 - contexttail
    sortidx = np.argsort(contexttail, kind = 'stable')
    
    contextsort[alt_idx:] = contexttail[sortidx] 
    allsorttrial = alltrial.copy()
    alltail = alltrial[alt_idx:]
    allsorttrial[alt_idx:] = alltail[sortidx]
    
    return contextsort, allsorttrial

def sortedtrialbyframe(rawtrialfil, trial, newskippedtrials):

    count = 0
    if len(newskippedtrials) > 0 :
        skippedrawtrialfil = rawtrialfil.copy()
        consec = count_consec(list(np.sort(newskippedtrials).astype(int)))

        for i, k in enumerate(consec):
            t = np.sort(newskippedtrials)[count] 

            if i == 0: 
                skippedrawtrialfil[rawtrialfil >= t] = skippedrawtrialfil[rawtrialfil >= t] + k
            else: 
                skippedrawtrialfil[skippedrawtrialfil >= t] = skippedrawtrialfil[skippedrawtrialfil >= t] + k
            count += k
    else:
        skippedrawtrialfil = trial.copy()
        
    return skippedrawtrialfil
    
""" Scripts for Loading Matlab Structs """

def loadmat_sbx(filename):
    """
    this function should be called instead of direct spio.loadmat

    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    """
    print(filename)
    data_ = scipy.io.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data_)


def _check_keys(dict):
    """
    checks if entries in dictionary rare mat-objects. If yes todict is called
    to change them to nested dictionaries
    """

    for key in dict:
        if isinstance(dict[key], scipy.io.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])

    return dict


def _todict(matobj):
    """
    A recursive function which constructs from matobjects nested dictionaries
    """

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, scipy.io.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_ca_mat(fname):
    """load results from cnmf"""

    ca_dat = {}
    try:
        with h5py.File(fname, 'r') as f:
            for k, v in f.items():
                try:
                    ca_dat[k] = np.array(v)
                except:
                    print(k + "not made into numpy array")
                    ca_dat[k] = v
    except:
        ca_dat = scipy.io.loadmat(fname)
        for key in ca_dat.keys():
            if isinstance(ca_dat[key], np.ndarray):
                ca_dat[key] = ca_dat[key].T
    return ca_dat