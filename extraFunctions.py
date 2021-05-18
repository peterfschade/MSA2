#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  7 14:06:40 2020

@author: livingstonelab
"""
import numpy as np
from scipy.signal import butter, lfilter
import csv
#import umap
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.fftpack import fft
from matplotlib import path


def gini(array):
    """Calculate the Gini coefficient of a numpy array."""
    # All values are treated equally, arrays must be 1d:
    array = array.flatten()
    if np.amin(array) < 0:
        # Values cannot be negative:
        array -= np.amin(array)
    # Values cannot be 0:
    array += 0.0000001
    # Values must be sorted:
    array = np.sort(array)
    # Index per array element:
    index = np.arange(1, array.shape[0] + 1)
    # Number of array elements:
    n = array.shape[0]
    # Gini coefficient:
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))


def MAD(data):
    '''from HeartPy:Python Heart Rate Analysis Toolkit
    computes median absolute deviation

    Function that compute median absolute deviation of data slice
    See: https://en.wikipedia.org/wiki/Median_absolute_deviation

    Parameters
    ----------
    data : 1-dimensional numpy array or list
        sequence containing data over which to compute the MAD

    Returns
    -------
    out : float
        the Median Absolute Deviation as computed

    Examples
    --------
    >>> x = [2, 4, 3, 4, 6, 7, 35, 2, 3, 4]
    >>> MAD(x)
    1.5
    '''
    med = np.median(data)
    return np.median(np.abs(data - med))


def hampel_filter(data, filtsize=6):
    '''Detect outliers based on hampel filter
    from HeartPy:Python Heart Rate Analysis Toolkit
    Funcion that detects outliers based on a hampel filter.
    The filter takes datapoint and six surrounding samples.
    Detect outliers based on being more than 3std from window mean.
    See:
    https://www.mathworks.com/help/signal/ref/hampel.html

    Parameters
    ----------
    data : 1d list or array
        list or array containing the data to be filtered

    filtsize : int
        the filter size expressed the number of datapoints
        taken surrounding the analysed datapoint. a filtsize
        of 6 means three datapoints on each side are taken.
        total filtersize is thus filtsize + 1 (datapoint evaluated)

    Returns
    -------
    out :  array containing filtered data

    Examples
    --------
    >>> from .datautils import get_data, load_exampledata
    >>> data, _ = load_exampledata(0)
    >>> filtered = hampel_filter(data, filtsize = 6)
    >>> print('%i, %i' %(data[1232], filtered[1232]))
    497, 496
    '''

    # generate second list to prevent overwriting first
    # cast as array to be sure, in case list is passed
    output = np.copy(np.asarray(data))
    onesided_filt = filtsize // 2
    for i in range(onesided_filt, len(data) - onesided_filt - 1):
        dataslice = output[i - onesided_filt: i + onesided_filt]
        mad = MAD(dataslice)
        median = np.median(dataslice)
        if output[i] > median + (3 * mad):
            output[i] = median
    return output


def find_msa_jumps(note, note_, t_):
    D = path.Path([(40000, 30000), (125000, 117000), (125000, 30000), (40000, 30000)])
    U = path.Path([(30000, 40000), (30000, 125000), (110000, 125000), (30000, 40000)])
    uj = U.contains_points(np.concatenate((note.reshape(-1, 1), note_.reshape(-1, 1)), axis=1))
    dj = D.contains_points(np.concatenate((note.reshape(-1, 1), note_.reshape(-1, 1)), axis=1))

    if any(uj) or any(dj):
        all_j = np.concatenate((np.tile('u', len(np.where(uj)[0])),
                                np.tile('d', len(np.where(dj)[0]))), axis=0)
        label_ind = np.concatenate((np.where(uj)[0], np.where(dj)[0]))
        ind_sort = np.argsort(label_ind).astype(int)
        sorted_ind = label_ind[ind_sort]
        label_ = all_j[ind_sort]
        strlabel = ''.join(map(str, label_))
    else:
        strlabel = 's'
        sorted_ind = np.array([0])

    # stemp = np.concatenate((np.sort(label_ind),np.array(len(note)).reshape(1,)))
    # notedur = np.diff(stemp)
    # falsej = np.diff(np.sort(label_ind)) < np.ceil(t_)

    return strlabel, sorted_ind


def butter_bandpass(d, lt, ht, fs, order=2):
    nyq = 0.5 * fs
    low = lt / nyq
    high = ht / nyq
    b, a = butter(order, [low, high], btype="band")
    y = lfilter(b, a, d)
    return y


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = csv.writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def add_column_in_csv(input_file, output_file, transform_row):
    """ Append a column in existing csv using csv.reader / csv.writer classes"""
    # Open the input_file in read mode and output_file in write mode
    with open(input_file, 'r') as read_obj, \
            open(output_file, 'w', newline='') as write_obj:
        # Create a csv.reader object from the input file object
        csv_reader = csv.reader(read_obj)
        # Create a csv.writer object from the output file object
        csv_writer = csv.writer(write_obj)
        # Read each row of the input csv file as list
        for row in csv_reader:
            # Pass the list / row in the transform function to add column text for this row
            transform_row(row, csv_reader.line_num)
            # Write the updated row / list to the output file
            csv_writer.writerow(row)


'''
def draw_umap(data, n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean', title=''):
    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric=metric
    )
    u = fit.fit_transform(data);
    fig = plt.figure()
    if n_components == 1:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], range(len(u)), c=data)
    if n_components == 2:
        ax = fig.add_subplot(111)
        ax.scatter(u[:, 0], u[:, 1], c=data)
    if n_components == 3:
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(u[:, 0], u[:, 1], u[:, 2], c=data, s=100)
    plt.title(title, fontsize=18)
'''


def pyAudspectrogram(signal, sampling_rate, window, step, plot=False,
                     show_progress=False):
    """
    Short-term FFT mag for spectogram estimation:
    Returns:
        a np array (numOfShortTermWindows x num_fft)
    ARGUMENTS:
        signal:         the input signal samples
        sampling_rate:  the sampling freq (in Hz)
        window:         the short-term window size (in samples)
        step:           the short-term window step (in samples)
        plot:           flag, 1 if results are to be ploted
        show_progress flag for showing progress using tqdm
    RETURNS:
    """
    window = int(window)
    step = int(step)
    signal = np.double(signal)
    signal = signal / (2.0 ** 15)
    dc_offset = signal.mean()
    maximum = (np.abs(signal)).max()
    signal = (signal - dc_offset) / (maximum - dc_offset)

    num_samples = len(signal)  # total number of signals
    count_fr = 0
    num_fft = int(window / 2)
    specgram = np.zeros((int((num_samples - step - window) / step) + 1, num_fft),
                        dtype=np.float64)
    for cur_p in tqdm(range(window, num_samples - step, step),
                      disable=not show_progress):
        count_fr += 1
        x = signal[cur_p:cur_p + window]
        X = abs(fft(x))
        X = X[0:num_fft]
        X = X / len(X)
        specgram[count_fr - 1, :] = X

    freq_axis = [float((f + 1) * sampling_rate) / (2 * num_fft)
                 for f in range(specgram.shape[1])]
    time_axis = [float(t * step) / sampling_rate
                 for t in range(specgram.shape[0])]

    if plot:
        fig, ax = plt.subplots()
        c = specgram.transpose()[::-1, :]
        imgplot = plt.imshow(c / c.max(axis=0))
        fstep = int(num_fft / 5.0)
        frequency_ticks = range(0, int(num_fft) + fstep, fstep)
        frequency_tick_labels = \
            [str(sampling_rate / 2 -
                 int((f * sampling_rate) / (2 * num_fft)))
             for f in frequency_ticks]
        ax.set_yticks(frequency_ticks)
        ax.set_yticklabels(frequency_tick_labels)
        t_step = int(count_fr / 3)
        time_ticks = range(0, count_fr, t_step)
        time_ticks_labels = \
            ['%.2f' % (float(t * step) / sampling_rate) for t in time_ticks]
        ax.set_xticks(time_ticks)
        ax.set_xticklabels(time_ticks_labels)
        ax.set_xlabel('time (secs)')
        ax.set_ylabel('freq (Hz)')
        imgplot.set_cmap('Greys')
        plt.colorbar()
        plt.show()

    return specgram, time_axis, freq_axis

# %%

# class vocal_dataset:

#     def __init__(self, species=None, ID=None, condition=None):
#         self.species = species
#         self.ID = ID
#         self.condition = condition

#     def extract_data(self, d=datafile ,wav=wavefile):
#         data_ = pd.read_csv(d)
#         wav_ = pd.read_csv(wav,sep='delimiter')
#         ind = data_['subj'].str.match(self.ID)
#         allcond = data_['cond'][ind].unique()
#         allsession = data_['session_no'][ind].unique()









