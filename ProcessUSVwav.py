#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:03:07 2020

@author: Peter Schade
"""

import numpy as np
from scipy.io import wavfile
from extraFunctions import gini, butter_bandpass,append_list_as_row,pyAudspectrogram
import os,csv
import time
from scipy import ndimage
from matplotlib import path


'''
put raw wavefiles into a 'raw_dir' directory
'''

maindir = 'C:/Users/JarvisLab/Desktop/AMVOCComparisons/MSA_Peter/DefaultVals/scrap'

savefile = 'MSA_Peter_amvocComparisons.csv'

rawdir = 'C:/Users/JarvisLab/Desktop/AMVOCComparisons/recordings'

sngdir = 'pyth'
savedir = maindir




clean = 0 # use cleaning procedure of MSA to compute features,not recommended
purity_band = 1000 # in Hz, for computing spec purity, width around dominant as peak
t_res = 2 # temporal resolution of spectogram
filter_w = 3 # median filter in samples ( multiply by t_res to get in ms)
snr_thresh = 10 #dB threshold above noise, 10dB is a factor of 10
purity_thresh = 0.4 # range:0-1, high pass
discnt_thresh = 1.3 # range:0-2, low pass
freq_thresh = 40e3 # in Hz, high pass
dura_thresh = 12 #in ms, high pass
merge_thresh = 10 # in ms, low merge syllables
note_thresh = 6 # in ms, high pass
f_hi = 110e3 # hi freq to consider
f_lo = 35e3 # low freq to consider

raw_listfiles = os.listdir(os.path.join(maindir,rawdir))
if os.path.isdir(os.path.join(maindir,sngdir)) == False:
    os.mkdir(os.path.join(maindir,sngdir))

#%% initialize text files
syllablefile = os.path.join(savedir,savefile)
with open(syllablefile, 'w+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
# =============================================================================
#     csv_writer.writerow(['file','syllableType','harmonic','duration',
#                          'ISI','fqVariance','purity','amplitude','fqmin','fqmean',
#                          'fqmax','fqstart','fqend','bandwidth'])
# =============================================================================

    csv_writer.writerow(['file','startTime','endTime','amplitude',
                          'stFq','EndFq','minFq','meanFq','maxFq','specPurity',
                          'cv','gini_c','giniSxx','vel', 'maxvel','minvel','jumpsize',
                          'SyllableType','ISI','startTime_sample','endTime_sample','voltage'])
write_obj.close()

unreadablefile = os.path.join(savedir,'unreadable.csv')
with open(unreadablefile, 'w+', newline='') as write_obj:
    csv_writer = csv.writer(write_obj)
    csv_writer.writerow(['file'])
write_obj.close()
#%%
session_no = 0
for ifile in raw_listfiles:
    _a_ = time.perf_counter()
    if ifile.lower().endswith('.wav') == False:
        append_list_as_row(unreadablefile,ifile)
        continue
     #%% read, bandpass filter, create spectrogram
    try:
        fs, data = wavfile.read(os.path.join(maindir,rawdir,ifile)) # read data
    except: 
        print('\n could not read file \n')
        append_list_as_row(unreadablefile,ifile)
        print(os.path.join(ifile))
        continue
        
    c = butter_bandpass(data,f_lo - 5e3,f_hi + 5e3,fs) # bandpass filter
    #f, t, Sxx = spectrogram(c, fs=fs, window='flattop',nperseg=512,noverlap=256, scaling='spectrum')
    #t_res = (t[1]-t[0])*1000
    Sxx, t, f = pyAudspectrogram(c, fs, fs*t_res/1000, fs*t_res/1000)
    Sxx = np.transpose(Sxx)
    f=np.array(f)
    t=np.array(t)
    temp = [ x for x, f in enumerate(f) if f <= f_lo or f >=f_hi]
    Sxx = np.delete(Sxx, temp,0)
    f = np.delete(f, temp,0)
    
    dt = t[1]-t[0]
    pur_band = np.ceil(purity_band/(f[1]-f[0])).astype(int)
    # save files
    spect_file = ifile.split('.')[0]+'.npz'
    np.savez(os.path.join(maindir,sngdir,spect_file), sng = Sxx, f=f, t=dt)
    param_file = ifile.split('.')[0]+'_params.npz'
    np.savez(os.path.join(maindir,sngdir,param_file),
             clean = clean,
             purity_band = pur_band, filter_w = filter_w,
             t_res = t_res, snr_thresh = snr_thresh,
             purity_thresh = purity_thresh, discnt_thresh = discnt_thresh,
             freq_thresh = freq_thresh, dura_thresh = dura_thresh,
             merge_thresh = merge_thresh, note_thresh = note_thresh,
             f_hi = f_hi, f_lo = f_lo
             )
    Sxx = ndimage.median_filter(Sxx, (2, filter_w))
    Sxx = pow(Sxx,2)
            
     #%%  calculate dB (ratio of signal to noise), threshold
     # 10dB change is a 10 fold change
    Sxx_ = Sxx - np.median(Sxx,axis=1).reshape(-1,1) # subtract out median at each freq
    ref_ = (np.median(abs(Sxx_), axis=1) / 0.6745).reshape(-1,1)
    _Sxx = 10 * np.log10((Sxx / ref_)) # in dB, relative to median of signal
    _Sxx[_Sxx <= 5] = 1e-15 # set ratio below 0 to 0
    Sxx_[_Sxx <= 5] = 1e-15
    tot_snr = _Sxx.sum(axis=0)
     #%% get time varying features of signal
    pow_ = pow(Sxx_,2) # calculate power (amplitude^2)
    tot_power = pow_.sum(axis=0) # total power of signal
    temp = np.argmax(pow_,axis=0)
    peak_power = np.zeros((len(tot_power),)) # peak power at each time point
    for i in range(pur_band,len(temp)-pur_band): # ends don't matter
        peak_power[i] = pow_[(temp[i]-pur_band):(temp[i]+pur_band),i].sum()
    del temp
    specpurity = peak_power/tot_power # calculate purity at each time point
    specpurity[np.isnan(specpurity)] = 0 # remove nans from divide by 0
    norm_power = (pow_ / tot_power) # normalized power
    norm_power[np.isnan(norm_power)] = 0 # remove nan from 0 power
    norm_sig = (_Sxx / tot_snr) # normalized signal
     
    norm_sig[np.isnan(norm_sig)] = 0 # remove nan from 0 power
    norm_sig = np.concatenate((np.zeros((len(f),1)),norm_power),axis=1) # add buffer for diff
    diff_freqpwr = abs(np.apply_along_axis(np.diff,1,norm_sig)) # derivative of frequency power
    spec_discnt = np.apply_along_axis(np.sum,0,diff_freqpwr) # spectral discontinuity
    #weighted_meanfreq = (norm_power*f.reshape(-1,1)).sum(axis=0) # weighted mean frequency
    weighted_meanfreq = f[norm_power.argmax(axis=0)]
     #%% extract syllables
     # find indexes that don't meet requirements
    failind_1= (weighted_meanfreq <= freq_thresh).reshape(1,-1)# index of failures
    failind_2 = (specpurity <= purity_thresh).reshape(1,-1) # index of failures
    failind_3 = (spec_discnt >= discnt_thresh).reshape(1,-1)# index of failures
    failind_4 = (tot_snr < snr_thresh).reshape(1,-1)# index of failures
    fail_ind_ = np.concatenate((failind_1,failind_2,failind_3,failind_4),axis=0).max(axis=0) # concatenate failures
    fail_ind = np.array([q for q,x in enumerate(fail_ind_) if x == True]).astype(int) # all failures
    longindx = np.array([q for q,x in enumerate(np.diff(fail_ind,1)) if x > np.ceil(note_thresh/t_res)]).astype(int) # find breaks in failures
    #%% merge syllables
    twhis = fail_ind[longindx[1:]] # initial
    twhis_ = fail_ind[longindx[1:]+1]
    dt_ = twhis[1:]-twhis_[0:-1] # time between extractions
    closei = np.array([q for q, x in enumerate(dt_) if x <= np.ceil(merge_thresh/t_res)]) #index of merges
    for i in range(len(closei)-1,0,-1): # merge back to front
        twhis_[closei[i]] = twhis_[closei[i]+1]
        twhis = np.delete(twhis,closei[i]+1)
        twhis_ = np.delete(twhis_,closei[i]+1)
     #%% check duration threshold, compute ISI
    dur_check = (twhis_-twhis) >= np.ceil(dura_thresh/t_res)
    twhis = twhis[dur_check]
    twhis_ = twhis_[dur_check]
    isi = np.concatenate((twhis[1:] - twhis_[:-1], [np.nan]))
     #%% time 
    __a__ = time.perf_counter() 
    print('\n nVocalizations:  ', str(len(twhis)), ' in ', ifile)
    print('processing time:', str(__a__ - _a_), 'seconds')
    print ('merge events: ', str(len(closei)))
     #%% get syllable information    
     #dwhis = (twhis_ - twhis) * (dt) # duration of whistle
    for inote in range(len(twhis)): 
        note = weighted_meanfreq[twhis[inote]:twhis_[inote]]
        noteind = np.array([q for q,x in enumerate(fail_ind_[twhis[inote]:twhis_[inote]]) if x == False]).astype(int)
        note_ = note[noteind]
        gini_c = gini(note_) # higher values -> more representation at single note
        giniSxx = np.apply_along_axis(gini,axis=0,arr=_Sxx[:,twhis[inote]:twhis_[inote]]).mean()
        cv = np.std(note_)/np.mean(note_) # coeff of variation
        vel = abs(np.diff(note_,1)).mean()
        maxvel = np.diff(note_,1).max()
        minvel = np.diff(note_,1).min()
        
        #MSA features below
        startfreq = note_[0]
        endfreq = note_[-1]
        minfreq = note_.min()
        maxfreq = note_.max()
        meanfreq = note_.mean()
        #MSA classification below
        _n = note_[:-1]
        df = np.diff(note_,1)
        n_ = df + _n
        D = path.Path([(40000,30000), (125000, 117000), (125000, 30000), (40000, 30000)])
        U = path.Path([(30000,40000), (30000, 125000), (110000, 125000), (30000, 40000)])
        uj = U.contains_points(np.concatenate((_n.reshape(-1,1),n_.reshape(-1,1)),axis = 1))
        dj = D.contains_points(np.concatenate((_n.reshape(-1,1),n_.reshape(-1,1)),axis = 1))
        falsej = []
        strlabel = ''
        if any(uj) or any(dj):
            all_j = np.concatenate((np.tile('u',len(np.where(uj)[0])),
                                    np.tile('d',len(np.where(dj)[0]))), axis=0)
            label_ind = np.concatenate((np.where(uj)[0],np.where(dj)[0]))
            all_j = np.delete(all_j,np.where(label_ind == 0)[0])
            label_ind = np.delete(label_ind,np.where(label_ind == 0)[0])
            
            ind_sort = np.argsort(label_ind).astype(int)
            sorted_ind = label_ind[ind_sort]
            label_ = all_j[ind_sort]
            strlabel = ''.join(map(str, label_))
            stemp = np.concatenate((np.sort(label_ind),np.array((len(_n)-1)).reshape(1,)))
            notedur = np.diff(stemp)
            stemp = stemp[:-1]
            falsej = np.where( notedur < np.ceil(dura_thresh/t_res))[0]
            jumpsize = abs(df[label_ind]).mean()
            count = 0
            while len(falsej) > 0 and falsej[-1] == (len(notedur)-1):
                f1 = stemp[falsej[-1]]-1
                f2 = len(_n)
                nullsamples = np.arange(f1,f2)
                _n = np.delete(_n, nullsamples,0)
                falsej = falsej[:-1]
            while len(falsej) > 0:
                count += 1
                print('.', end='')
                falsejr = falsej[::-1]
                for fj in falsejr:
                    fsampstart = stemp[fj]
                    fsampend = stemp[fj+1]
                    nullsamples = np.arange(fsampstart, fsampend+1)
                    if fsampend < [len(_n)-1]:
                        if fsampstart == 1:
                            fsampstart = 2
                        _n[nullsamples] = _n[fsampend+1]
                    
                _n_ = np.concatenate((np.diff(_n), [0]))
                n_= _n + _n_
                uj = U.contains_points(np.concatenate((_n.reshape(-1,1),n_.reshape(-1,1)),axis = 1))
                dj = D.contains_points(np.concatenate((_n.reshape(-1,1),n_.reshape(-1,1)),axis = 1))
                all_j = np.concatenate((np.tile('u',len(np.where(uj)[0])),
                                    np.tile('d',len(np.where(dj)[0]))), axis=0)
                label_ind = np.concatenate((np.where(uj)[0],np.where(dj)[0]))
                all_j = np.delete(all_j,np.where(label_ind == 0)[0])
                label_ind = np.delete(label_ind,np.where(label_ind == 0)[0])
                ind_sort = np.argsort(label_ind).astype(int)
                sorted_ind = label_ind[ind_sort]
                label_ = all_j[ind_sort]
                strlabel = ''.join(map(str, label_))
                stemp = np.sort(label_ind)
                notedur = np.diff(stemp)
                falsej = np.where( notedur <  np.ceil(dura_thresh/t_res))[0]
                jumpsize = abs(_n_[label_ind]).mean()
        if len(falsej) == 0 and len(strlabel) == 0:
            strlabel = 's'
            sorted_ind = np.array([0])
            jumpsize = 0
        if len(_n) < np.ceil(note_thresh/t_res): # changed from continue 10/02/2020
            strlabel = 'notIDd'
        if clean == 1: #from MSA, mostly legacy, not recommended
            gini_c = gini(_n) # higher values -> more representation at single note
            giniSxx = np.apply_along_axis(gini,axis=0,arr=_Sxx[:,twhis[inote]:twhis_[inote]]).mean()
            cv = np.std(_n)/np.mean(_n) # coeff of variation
            vel = abs(np.diff(_n,1)).mean()
            maxvel = np.diff(_n,1).max() 
            minvel = np.diff(_n,1).min()
            #MSA features below
            startfreq = _n[0]
            endfreq = _n[-1]
            minfreq = _n.min()
            maxfreq = _n.max()
            meanfreq = _n.mean()
        
        spec_pur = specpurity[twhis[inote]:twhis_[inote]][noteind].mean() # how 'pure' is it
        amplitude = tot_snr[twhis[inote]:twhis_[inote]][noteind].mean()
        syldur = (twhis_[inote]*dt) - (twhis[inote]*dt)
        fqbandwidth = maxfreq-minfreq
        voltage = np.abs(data[((twhis[inote]+1)*dt*fs).astype(int):((twhis_[inote]+1)*dt*fs).astype(int)]).mean()
        towrite = [ifile[0:len(ifile)-4],twhis[inote]*dt,twhis_[inote]*dt,
                   amplitude,startfreq,endfreq,minfreq,meanfreq,maxfreq,spec_pur,
                    cv,gini_c,giniSxx,vel,maxvel,minvel,jumpsize,strlabel,isi[inote]*dt,
                    twhis[inote],twhis_[inote],voltage]
        towrite_ = [ifile[0:len(ifile)-4], strlabel, 0, syldur, isi[inote]*dt, cv, spec_pur, 
                   amplitude, minfreq, meanfreq, maxfreq, startfreq, endfreq, fqbandwidth]
                   
        append_list_as_row(syllablefile, towrite) # write
     
#%% plotting of whistles
# for inote in range(1,50): #range(len(twhis)):
#      plt.figure()
#      c_ = _Sxx[:,twhis[inote]:twhis_[inote]]
#      c_ = np.transpose(z_)
#      c_ = c_ / c_.sum(axis=0)
# # # # # #  #     axes[0].plot(pow(tot_snr[twhis[i]:twhis_[i]],2))
#    #  plt.pcolormesh(t[np.arange((twhis_[inote]) - (twhis[inote]))], f, c_,cmap='gray_r')
#      plt.pcolormesh(t[np.arange(13)], t[np.arange(250)], c_,cmap='gray_r')
# #     plt.pause(100)
    
#%%
# plt.figure()
# plt.plot(n_, 'k.')
# plt.pause(100)
# plt.plot(twhis,np.zeros(len(twhis)), 'r.')
# plt.plot(twhis_,np.zeros(len(twhis)), 'b.')
#%%
# nn=0
# for ifile in raw_listfiles:
#     if ifile.lower().endswith('.wav') == False:
#         continue
#     try:
#         fs, data = wavfile.read(os.path.join(maindir,rawdir,ifile)) # read data
#     except: 
#         print('could not read file \n')
#         print(os.path.join(maindir,rawdir,ifile))
#         append_list_as_row(unreadablefile,ifile)
#         nn+=1
#         continue

