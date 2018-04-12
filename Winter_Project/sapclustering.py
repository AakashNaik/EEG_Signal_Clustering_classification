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
import numpy as np
from scipy.stats import kurtosis, skew

channels = ['Fp1', 'AFp1', 'Fpz', 'AFp2', 'Fp2', 'AF7', 'AF3', 'AF4', 'AF8', 'FAF5', 'FAF1', 'FAF2', 'FAF6',
                'F7', 'F5', 'F3', 'F1', 'Fz', 'F2', 'F4', 'F6', 'F8', 'FFC7', 'FFC5', 'FFC3', 'FFC1', 'FFC2', 'FFC4',
                 'FFC6', 'FFC8', 'FT9', 'FT7', 'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6', 'FT8', 'FT10', 'CFC7',
                 'CFC5', 'CFC3', 'CFC1', 'CFC2', 'CFC4', 'CFC6', 'CFC8', 'T7', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6'
                 , 'T8', 'CCP7', 'CCP5', 'CCP3', 'CCP1', 'CCP2', 'CCP4', 'CCP6', 'CCP8', 'TP9', 'TP7', 'CP5', 'CP3', 'CP1',
                 'CPz', 'CP2', 'CP4', 'CP6', 'TP8', 'TP10', 'PCP7', 'PCP5', 'PCP3', 'PCP1', 'PCP2', 'PCP4', 'PCP6', 'PCP8',
                 'P9', 'P7', 'P5', 'P3', 'P1', 'Pz', 'P2', 'P4', 'P6', 'P8', 'P10', 'PPO7', 'PPO5', 'PPO1', 'PPO2', 'PPO6',
                 'PPO8', 'PO7', 'PO3', 'PO1', 'POz', 'PO2', 'PO4', 'PO8', 'OPO1', 'OPO2', 'O1', 'Oz', 'O2', 'OI1', 'OI2',
                 'I1', 'I2']

main_channels=["FC2", "FC4", "FC6", "CFC2", "CFC4", "CFC6", "C2", "C4", "C6", "CCP2", "CCP4", "CCP6", "CP2", "CP4", "CP6", "FC5", "FC3", "FC1", "CFC5", "CFC3", "CFC1", "C5", "C3", "C1", "CCP5", "CCP3", "CCP1", "CP5", "CP3" ,"CP1"]


Training_data=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_cnt.txt'
Reduced_channel=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/signal_few_channels.txt'
markers=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt'
## print len(channels)
signal_array = np.loadtxt(Training_data)
signal_channel = np.loadtxt(Reduced_channel)     #data taken from data file  data consist of 280 trials(140 for each)*118 channels
b, a = signal.butter(3, np.array([7, 30])/ 100.00, 'bandpass')

## ## ## print signal_array.shape[0]
## ## print signal_array.shape[1]  #found the butterworth filter coefficient
signal_array1 = signal.lfilter(b, a, signal_array, axis = -1)
signal_channel1 = signal.lfilter(b, a, signal_channel, axis = -1)

## ## print signal_array1.shape[0]
## ## print signal_array1.shape[1]      #applied the filter
marker_array = [map(str,l.split('\t')) for l in open(markers).readlines()] #splitted up the marker string which consist of row number and it's data class
### ## print marker_array
time = np.arange(len(signal_array1))
time1 = np.arange(len(signal_channel1))        #changed signal_array to signal_array1 created time array where signal_array can be uniformally distributed ,signal array is the original data file not filtered one
train_markers1=[]
train_markers1 = [(float(events[0]),str(events[1])) for events in marker_array if events[1]!= '0\n']   #choosing train markers neglecting whose class is marked as 0
for events in marker_array:
    if events[1] != '0':
        train_markers1.append((float(events[0]) + 100.0, str(events[1])))              #taking only middle elements between 100&200
        train_markers1.append((float(events[0]) + 200.0, str(events[1])))                #again filling up the data this time taking two rows for same data point one is starting point other is end point
markers1 = np.array(train_markers1)       #markers1 is numpy array of train_markers1
markers_subject1_class_1 = [(float(events[0]),str(events[1])) for events in markers1 if events[1] == '1\n']    #  separate line index starting and ending  for class 1 and class 2
markers_subject1_class_2 = [(float(events[0]),str(events[1])) for events in markers1 if events[1] == '2\n']     #markers_subject1_class_1 and like wise other class correspondingly contains the 1 and 2 class marker data points
## ## print type(markers_subject1_class_1),type(markers_subject1_class_2),"markers_subject1_class1&2 data type"
cnt1 = convert_mushu_data(signal_array1, markers_subject1_class_1,118,channels)    #convert data into continuous form   for 1st and second classs
cnt2 = convert_mushu_data(signal_array1, markers_subject1_class_2,118,channels)
cnt_ch1 = convert_mushu_data(signal_channel1, markers_subject1_class_1,30,main_channels)    #convert data into continuous form   for 1st and second classs
cnt_ch2 = convert_mushu_data(signal_channel1, markers_subject1_class_2,30,main_channels)
## ## print cnt1,"cnt1 shape"      #What type of marker data should be there  should it contain start as well as end point  or just start point is required

md = {'class 1': ['1\n'],'class 2': ['2\n']}

epoch_subject1_class1 = segment_dat(cnt1, md, [0, 1000])        #epoch is a 3-d data set  class*time*channel
epoch_subject1_class2 = segment_dat(cnt2, md, [0, 1000])
epoch_subject1_ch1_class1 = segment_dat(cnt_ch1, md, [0, 1000])        #epoch is a 3-d data set  class*time*channel
epoch_subject1_ch1_class2 = segment_dat(cnt_ch2, md, [0, 1000])

### ## print "epoch data",epoch_subject1_class1
def bandpowers(segment):
     features = []
     ## ## print len(segment),"segment jbjb"
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


final_epoch1 = append_epo(epoch_subject1_class1, epoch_subject1_class2)
final_epoch_ch1 = append_epo(epoch_subject1_ch1_class1, epoch_subject1_ch1_class2)          #appended both the epoch data sets
w1,a1,d1 = calculate_csp(final_epoch1)
w2,a2,d2 = calculate_csp(final_epoch_ch1)                                #calculate csp   but why we need to append the data and calculate the csp paramters waht if we calculate it individually
fil_epoch_subject1_class1 = apply_csp(epoch_subject1_class1, w1, [0,1,2,3,4,-5,-4,-3,-2,-1])     # brackets number are the column number to use
fil_epoch_subject1_class2 = apply_csp(epoch_subject1_class2, w1, [0,1,2,3,4,-5,-4,-3,-2,-1])
fil_final_epoch1 = append_epo(fil_epoch_subject1_class1, fil_epoch_subject1_class2)    # final filtered epo     class*time*channel
fil_epoch_subject1_ch1_class1 = apply_csp(epoch_subject1_ch1_class1, w2, [0,1,2,3,4,-5,-4,-3,-2,-1])     # brackets number are the column number to use
fil_epoch_subject1_ch1_class2 = apply_csp(epoch_subject1_ch1_class2, w2, [0,1,2,3,4,-5,-4,-3,-2,-1])
fil_final_epoch_ch1 = append_epo(fil_epoch_subject1_class1, fil_epoch_subject1_class2)    # final filtered epo     class*time*channel

## ## print "dddd"
## ## print fil_epoch_subject1_class1.data.shape
## ## print fil_epoch_subject1_class2.data.shape
## ## print fil_final_epoch1
data=copy.copy(fil_final_epoch1.data)
data2=copy.copy(fil_final_epoch_ch1.data)

targets = fil_final_epoch1.axes[0]
targets2 = fil_final_epoch_ch1.axes[0]
## ## print "sorrow"
## ## print data,"breaking now"
## print targets
## ## print fil_final_epoch1.data[0],"epoch data"
data=np.array(data)
data2=np.array(data2)
mean_data=[]
## print data.shape[0],data.shape[1],data.shape[2],"not for long"
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        summ=0
        for k in range(data.shape[2]):
            summ+=data[i,j,k]
        array2.append(summ/data.shape[1])
    ## ## print "array",array2
    mean_data.append(array2)
mean_data=np.array(mean_data)
##### print array.shape[0],array.shape[1],"ffff"

std=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        summ=0
        array2.append(np.std(data[i,j,0:]))
    ## ## print "array",array2
    std.append(array2)

skew1=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        summ=0
        #print"skkkew"
        x=skew(data[i,j,0:])
        array2.append(x.real)
    ## ## print "array",array2
    skew1.append(array2)

max_data=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        summ=0
        array2.append(np.amax((data[i,j,0:])))
    ## ## print "array",array2
    max_data.append(array2)

kurtosis1=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        y=kurtosis(data[i,j,0:])
        array2.append(y.real)
    ## ## print "array",array2
    kurtosis1.append(array2)

mean_data_ch1=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        summ=0
        for k in range(data.shape[2]):
            summ+=data[i,j,k]
        array2.append(summ/data.shape[1])
    ## ## print "array",array2
    mean_data_ch1.append(array2)
mean_data_ch1=np.array(mean_data)
##### print array.shape[0],array.shape[1],"ffff"

std_ch1=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        summ=0
        array2.append(np.std(data[i,j,0:]))
    ## ## print "array",array2
    std_ch1.append(array2)

skew_ch1=[]
for i in range(data.shape[0]):
    array2=[]
    for j in range(data.shape[1]):
        summ=0
        #print"skkkew"
        x=skew(data[i,j,0:])
        array2.append(x.real)
    ## ## print "array",array2
    skew_ch1.append(array2)

max_data_ch1=[]
for i in range(data2.shape[0]):
    array2=[]
    for j in range(data2.shape[1]):
        summ=0
        array2.append(np.amax((data2[i,j,0:])))
    ## ## print "array",array2
    max_data_ch1.append(array2)

kurtosis_ch1=[]
for i in range(data2.shape[0]):
    array2=[]
    for j in range(data2.shape[1]):
        y=kurtosis(data2[i,j,0:])
        array2.append(y.real)
    ## ## print "array",array2
    kurtosis_ch1.append(array2)

#print(kurtosis1)

'''kf = KFold(n_splits=10,random_state = 30, shuffle = True)     #splitting into 10
  kf.get_n_splits(array)
for train_index, test_index in kf.split(array):
    X_train, X_test = array[train_index],array[test_index]
    y_train, y_test = targets[train_index], targets[test_index]
    experiment=svm.SVC()
    targets=np.array(targets)
    experiment.fit(X_test,y_test)
    count=0
    total=len(y_test)
    #for x,y  in zip(X_train,y_train):
    predict=[]
    predict=experiment.predict(X_test)
    for x in range(len(predict)) :
        if y_test[x]==predict[x] :
            count+=1
    ### ## print predict        #count+=1
    ## ## print count
    ## ## ## print len(predict)
    ## ## print float(count)/len(predict)*100
'''


dictionary1 = []
dictionary2 = []
dictionary_ch1=[]
dictionary_ch2=[]
count=0
## ## print len(fil_final_epoch1.axes[0]),"fil_final_epoch1.axes"
for i in range(len(fil_final_epoch1.axes[0])):
    ## ## print count
    count+=1              #what fil_final_epoch1.axes[0]will return?
    segment = fil_final_epoch1.data[i]
    ## ## print segment,"segmenttt"
    segment = np.array(segment)
    segment = np.transpose(segment)

    features1 = bandpowers(segment)                       #calculating bandpower for a segment
    features2 = wavelet_features(segment)                 #calculating wavelet_features for a segment

    dictionary1.append(features1)                         #dictionary 1 contains bandpowers
    dictionary2.append(features2)                         #dictionary 2 contains wavrlet_features


dictionary1 = np.array(dictionary1)    #dictionary 1 contain bandpower features
dictionary2 = np.array(dictionary2)

for i in range(len(fil_final_epoch_ch1.axes[0])):
    ## ## print count
    count+=1              #what fil_final_epoch1.axes[0]will return?
    segment1 = fil_final_epoch_ch1.data[i]
    ## ## print segment,"segmenttt"
    segment1 = np.array(segment1)
    segment1 = np.transpose(segment1)

    features1 = bandpowers(segment1)                       #calculating bandpower for a segment
    features2 = wavelet_features(segment1)                 #calculating wavelet_features for a segment

    dictionary_ch1.append(features1)                         #dictionary 1 contains bandpowers
    dictionary_ch2.append(features2)                         #dictionary 2 contains wavrlet_features


dictionary_ch1 = np.array(dictionary1)    #dictionary 1 contain bandpower features
dictionary_ch2 = np.array(dictionary2)

'''## ## print "dictionary shape",dictionary1.shape[0]
## ## print dictionary1.shape[1],"dictionary matrix shape"
dictionary_bandpower  = dictionary1
dictionary_wavelet = dictionary2
## ## print fil_final_epoch1.axes[0],"sfsf"

targets = fil_final_epoch1.axes[0]
## ## print "target data"
## ## print type(targets)
dictionary_bandpower.shape
### ## print dictionary1,"dictionary1",dictionary1.shape[0],dictionary1.shape[1],dictionary1.shape[2]

from scipy.sparse import coo_matrix
X_sparse = coo_matrix(dictionary_wavelet)            # create sparse matrix in COOrdinate form

from sklearn.utils import resample
dictionary_wavelet, X_sparse, ywe = resample(dictionary_wavelet, X_sparse, targets, random_state=0)

from scipy.sparse import coo_matrix
X_sparse = coo_matrix(dictionary_bandpower)'''

#from sklearn.utils import resample
#dictionary_bandpower, X_sparse, ybp = resample(dictionary_bandpower, X_sparse, targets, random_state=0)


from sklearn.model_selection import KFold
## ## print "kmeans"
print "mean"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(mean_data)
labels = km.labels_
print(labels)

'''print "std"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(std)
labels = km.labels_
print(labels)

print "max_data"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(max_data)
labels = km.labels_
print(labels)

print "skew"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(skew1)
labels = km.labels_
print(labels)

print "kurtosis"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(kurtosis1)
labels = km.labels_
print(labels)'''

print"mean"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(mean_data)
labels=sc.labels_
print(labels)

print "std"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(std)
labels=sc.labels_
print(labels)

print "max_data"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(max_data)
labels=sc.labels_
print(labels)

print "skew"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(skew1)
labels=sc.labels_
print(labels)

print "kurtosis"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(kurtosis1)
labels=sc.labels_
print(labels)

print "bandpower using spectral clustering"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(dictionary1)
labels=sc.labels_
print(labels)

print "wavelet features using spectral clustering"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(dictionary2)
labels=sc.labels_
print(labels)


from sklearn.model_selection import KFold
## ## print "kmeans"
print "mean"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(mean_data_ch1)
labels = km.labels_
print(labels)

print "std"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(std_ch1)
labels = km.labels_
print(labels)

print "max_data"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(max_data_ch1)
labels = km.labels_
print(labels)

print "skew"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(skew_ch1)
labels = km.labels_
print(labels)

print "kurtosis"
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(kurtosis_ch1)
labels = km.labels_
print(labels)

print"mean"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(mean_data_ch1)
labels=sc.labels_
print(labels)

print "std"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(std_ch1)
labels=sc.labels_
print(labels)

print "max_data"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(max_data_ch1)
labels=sc.labels_
print(labels)

print "skew"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(skew_ch1)
labels=sc.labels_
print(labels)

print "kurtosis"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(kurtosis_ch1)
labels=sc.labels_
print(labels)

print "bandpower using spectral clustering"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(dictionary_ch1)
labels=sc.labels_
print(labels)

print "wavelet features using spectral clustering"
sc = SpectralClustering(2, affinity='rbf', assign_labels='discretize',n_init=100)
sc.fit(dictionary_ch2)
labels=sc.labels_
print(labels)

'''km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(dictionary_wavelet)
labels = km.labels_
## ## print(labels)

mkm = MiniBatchKMeans(n_clusters=2,max_iter = 5000,n_init = 50)
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
## ## print(labels)

mkm = SpectralClustering(n_clusters=2)
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
## ## print(labels)

mkm = MeanShift()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
## ## print(labels)

mkm = AffinityPropagation()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
## ## print(labels)

mkm = AgglomerativeClustering(n_clusters =2)
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
## ## print(labels)

mkm = DBSCAN()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
## ## print(labels)

mkm = Birch()
mkm.fit(dictionary_wavelet)
labels = mkm.labels_
## ## print(labels)

km = KMeans(n_clusters=2,max_iter = 5000,n_init = 50)
km.fit(dictionary_bandpower)
labels = km.labels_
## ## print(labels)

mkm = MiniBatchKMeans(n_clusters=2,max_iter = 5000,n_init = 50)
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
## ## print(labels)

mkm = SpectralClustering(n_clusters=2)
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
## ## print(labels)

mkm = MeanShift()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
## ## print(labels)

mkm = AffinityPropagation()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
## ## print(labels)

mkm = AgglomerativeClustering(n_clusters =2)
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
## ## print(labels)

mkm = DBSCAN()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
## ## print(labels)

mkm = Birch()
mkm.fit(dictionary_bandpower)
labels = mkm.labels_
## ## print(labels)
'''
### ## print "circling"
'''kf = KFold(n_splits=10,random_state = 30, shuffle = True)     #splitting into 10
kf.get_n_splits(dictionary_wavelet)
## ## print "vultures"
y_classifier1 = []
y_classifier2 = []
y_classifier3 = []
y_classifier4 = []

y_all1 = []
_all2 = []
y_all3 = []
y_all4 = []
y_example_test = []
y_final_test = []

## ## print(kf)


def calculate_accuracy(Xts,yts,D,class1,class2,n):
    y_pred1 = []
    y_pred2 = []
    y_pred3 = []
    y_pred4 = []
    diff = []
    counter1 = 0
    counter2 = 0
    for i in range(len(Xts)):
        features = Xts[i]

        omp = OrthogonalMatchingPursuit(n_nonzero_coefs=10)
        omp.fit(D,features)
        coef = omp.coef_
        p = 0
        q = 0
        l = 0
        m = 0
        a = 0
        b = 0

        list1 = coef[0:min(class1,class2)]
        list2 = coef[min(class1,class2)+1:2*min(class1,class2)]

        c1 = (sum(z*z for z in list1))**(1/2.0)
        c2 = (sum(z*z for z in list2))**(1/2.0)

        p = np.std(list1)
        q = np.std(list2)

        a = max(list1)
        b = max(list2)

        for ko in range(min(class1,class2)):
            l = l + coef[ko]

        for io in xrange(min(class1,class2)+1,2*min(class1,class2)):
            m = m + coef[io]

        if p > q:
            y_pred1.append(0)

        else:
            y_pred1.append(1)
            if(yts[i] != 0):
                if(counter1==0):
                    counter1 +=1
                    idx_r, = coef.nonzero()
                    plt.xlim(0, len(coef))
                    y_pred3.append(1)
                    plt.title("Sparse Signal")
                    plt.stem(idx_r, coef[idx_r])
                    plt.show()

        if l>m:
            y_pred2.append(0)
        else:
            y_pred2.append(1)

        if a>b:
            y_pred3.append(0)
        else:
            y_pred3.append(1)
        if c1 > c2:
            y_pred4.append(0)
        else:
            y_pred4.append(1)


    ## ## print('class1', metrics.accuracy_score(yts, y_pred4, normalize=True, sample_weight=None))

    ## ## print('class2', metrics.accuracy_score(yts, y_pred3, normalize=True, sample_weight=None))

    ## ## print('class3', metrics.accuracy_score(yts, y_pred1, normalize=True, sample_weight=None))

    ## ## print('class4',metrics.accuracy_score(yts, y_pred2, normalize=True, sample_weight=None))

    ## ## print('\\n')


    return y_pred4,y_pred3,y_pred1,y_pred2

for train_index, test_index in kf.split(dictionary):
    ## ## print "a"
    X_train, X_test = dictionary[train_index], dictionary[test_index]
    y_train, y_test = ywe[train_index], ywe[test_index]
    class1 = 0
    class2 = 0
    for i in range(len(y_train)):
        if(y_train[i] == 0):
            class1 += 1;
            ## ## print "b"
        else:
            class2 += 1;
            ## ## print "c"
    reb_y = []
    reb_dic = []
    count = 0
    iterator = 0
    while(count < min(class1,class2)):
        if y_train[iterator] == 0:
            reb_dic.append(X_train[iterator])
            reb_y.append(0)
            count += 1
        iterator += 1
        ## ## print "d"

    count = 0
    iterator = 0
    while(count < min(class1,class2)):
        if y_train[iterator] == 1:
            reb_dic.append(X_train[iterator])
            reb_y.append(1)
            count += 1
        iterator += 1
        ## ## print "f" ,iterator
    reb_dictionary = np.array(reb_dic)
    reb_dictionary = reb_dictionary.transpose()
    y_classifier1,y_classifier2,y_classifier3,y_classifier4 = calculate_accuracy(X_test,y_test,reb_dictionary,class1,class2,5)
    ## ## print "g"
    y_all1.extend(y_classifier1);
    y_all2.extend(y_classifier2);
    y_all3.extend(y_classifier3);
    y_all4.extend(y_classifier4);

    y_final_test.extend(y_test);
    y_example_test = y_test
    ## ## print "h"'''
