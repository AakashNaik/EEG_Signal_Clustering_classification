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

main_channels=['FC2', 'FC4', 'FC6', 'CFC2', 'CFC4', 'CFC6', 'C2', 'C4', 'C6', 'CCP2', 'CCP4', 'CCP6', 'CP2', 'CP4', 'CP6', 'FC5', 'FC3', 'FC1', 'CFC5', 'CFC3', 'CFC1', 'C5', 'C3', 'C1', 'CCP5', 'CCP3', 'CCP1', 'CP5', 'CP3' ,'CP1']


Training_data=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_cnt.txt'
markers=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt'
## print len(channels)
signal_array = np.loadtxt(Training_data)     #data taken from data file  data consist of 280 trials(140 for each)*118 channels

main_channels_index=[]
for i in range(len(main_channels)):
    for j in range(len(channels)):
        if main_channels[i]==channels[j]:
            main_channels_index.append(j)
k=0
print main_channels_index
signal_few_channels=[]
for i in  range(signal_array.shape[0]) :
    array=[]
    for j in main_channels_index :

        array.append(signal_array[i][j])
    signal_few_channels.append(array)


array=[]
for i in  range(1,31):
    array.append(i)

signal_few_channels=pd.DataFrame(signal_few_channels)
signal_few_channels.columns=array
signal_few_channels.to_csv(r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/signal_few_channels.txt', header=None, index=None, sep=' ', mode='w')
