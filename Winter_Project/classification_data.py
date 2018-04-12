import numpy as np
import copy
#import pyedflib
from matplotlib import pyplot as plt
from nitime import utils
from nitime import algorithms as alg
from nitime.timeseries import TimeSeries
from nitime.viz import plot_tseries
import csv
import pywt
import scipy.stats as sp
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
#from sklearn.lda import LDA
from scipy import signal
from os import listdir
from os.path import isfile, join
from wyrm import processing as proc
from wyrm.types import Data
from wyrm.io import convert_mushu_data
from sklearn import metrics
from wyrm.processing import calculate_csp,segment_dat,apply_csp,append_epo
from wyrm.processing import select_channels
from wyrm.processing import swapaxes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import OrthogonalMatchingPursuitCV
from sklearn.preprocessing import normalize
import pandas as pd
from sklearn.cluster import MiniBatchKMeans,KMeans,SpectralClustering,MeanShift,AffinityPropagation,AgglomerativeClustering,DBSCAN,Birch
from sklearn import svm
from sklearn.model_selection import KFold

channels = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4',
                 'FFC6', 'FFC8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7',
                 'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'
                 , 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8',
                 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1', 'PPO2', 'PPO6',
                 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1', 'Oz', 'O2', 'OI1', 'OI2',
                 'I1', 'I2']

Training_data=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_cnt.txt'
markers=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt'

signal_array = np.loadtxt(Training_data)     #data taken from data file
b, a = signal.butter(3, np.array([7, 30])/ 100.00, 'bandpass')
print signal_array,"done"
print signal_array.shape[0]
print signal_array.shape[1]  #found the butterworth filter coefficient
signal_array1 = signal.lfilter(b, a, signal_array, axis = -1)
print signal_array1,"dooone"
print signal_array1.shape[0]
print signal_array1.shape[1]      #applied the filter
marker_array = [map(str,l.split('\t')) for l in open(markers).readlines()]     #splitted up the marker string which consist of row number and it's data class
#print marker_array
time = np.arange(len(signal_array))        #created time array where signal_array can be uniformally distributed ,signal array is the original data file not filtered one
train_markers1=[]
train_markers1 = [(float(events[0]),str(events[1])) for events in marker_array if events[1]!= '0\n']   #choosing train markers neglecting whose class is marked as 0
for events in marker_array:
    if events[1] != '0':
        train_markers1.append((float(events[0]) + 100.0, str(events[1])))              #taking only middle elements between 100&200
        train_markers1.append((float(events[0]) + 200.0, str(events[1])))                #again filling up the data this time taking two rows for same data point one is starting point other is end point
markers1 = np.array(train_markers1)       #markers1 is numpy array of train_markers1
markers_subject1_class_1 = [(float(events[0]),str(events[1])) for events in markers1 if events[1] == '1\n']    #  separate line index starting and ending  for class 1 and class 2
markers_subject1_class_2 = [(float(events[0]),str(events[1])) for events in markers1 if events[1] == '2\n']     #markers_subject1_class_1 and like wise other class correspondingly contains the 1 and 2 class marker data points
#print "cerberus",markers_subject1_class_1
cnt1 = convert_mushu_data(signal_array, markers_subject1_class_1,100,channels)    #convert data into continuous form   for 1st and second classs
cnt2 = convert_mushu_data(signal_array, markers_subject1_class_2,100,channels)
#print cnt1,"cnt1 shape"      #What type of marker data should be there  should it contain start as well as end point  or just start point is required

md = {'class 1': ['1\n'],'class 2': ['2\n']}

epoch_subject1_class1 = segment_dat(cnt1, md, [0, 1000])        #epoch is a 3-d data set  class*time*channel
epoch_subject1_class2 = segment_dat(cnt2, md, [0, 1000])

#print "epoch data",epoch_subject1_class1
def bandpowers(segment):
     features = []
     for i in range(len(segment)):
         f,Psd = signal.welch(segment[i,:], 100)
         power1 = 0
         power2 = 0
         f1 = []
         for j in range(0,len(f)):
             if(f[j]>=4 and f[j]<=13):
                 power1 += Psd[j]
             if(f[j]>=14 and f[j]<=30):
                 power2 += Psd[j]
         features.append(power1)
         features.append(power2)
     return features


def wavelet_features(epoch):     #implementation of wavelet features
    cA_values = []
    cD_values = []
    cA_mean = []
    cA_std = []
    cA_Energy =[]
    cD_mean = []
    cD_std = []
    cD_Energy = []
    Entropy_D = []
    Entropy_A = []
    features = []
    for i in range(len(epoch)):
        cA,cD=pywt.dwt(epoch[i,:],'coif1')
        cA_values.append(cA)
        cD_values.append(cD)#calculating the coefficients of wavelet transform.
    for x in range(len(epoch)):
        cA_Energy.append(abs(np.sum(np.square(cA_values[x]))))
        features.append(abs(np.sum(np.square(cA_values[x]))))

    for x in range(len(epoch)):
        cD_Energy.append(abs(np.sum(np.square(cD_values[x]))))
        features.append(abs(np.sum(np.square(cD_values[x]))))

    return features


final_epoch1 = append_epo(epoch_subject1_class1, epoch_subject1_class2)          #appended both the epoch data sets
w1,a1,d1 = calculate_csp(final_epoch1)                                #calculate csp   but why we need to append the data and calculate the csp paramters waht if we calculate it individually
fil_epoch_subject1_class1 = apply_csp(epoch_subject1_class1, w1, [0,1,2,3,4,-5,-4,-3,-2,-1])     # brackets number are the column number to use
fil_epoch_subject1_class2 = apply_csp(epoch_subject1_class2, w1, [0,1,2,3,4,-5,-4,-3,-2,-1])
fil_final_epoch1 = append_epo(fil_epoch_subject1_class1, fil_epoch_subject1_class2)    # final filtered epo     class*time*channel
'''print "dddd"
print fil_epoch_subject1_class1.data.shape
print fil_epoch_subject1_class2.data.shape
print fil_final_epoch1'''
data=copy.copy(fil_final_epoch1.data)

targets = fil_final_epoch1.axes[0]
print "sorrow"
print data
print targets
data=np.array(data)
array=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[2]):
        summ=0
        for k in range(data.shape[1]):
            summ+=data[i,k,j]
        array2.append(summ/data.shape[1])
    print "array",array2
    array.append(array2)
array=np.array(array)
array=pd.DataFrame(array)
targets=pd.DataFrame(targets)
array.to_csv(r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/clustering_data.txt', header=None, index=None, sep=' ', mode='w')
targets.to_csv(r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/target_data.txt', header=None, index=None, sep=' ', mode='w')
