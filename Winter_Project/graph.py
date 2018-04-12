
# The %... is an iPython thing, and is not part of the Python language.
# In this case we're just telling the plotting library to draw things on
# the notebook, instead of on a separate window.

#this line above prepares IPython notebook for working with matplotlib

# See all the "as ..." contructs? They're just aliasing the package names.
# That way we can call methods like plt.plot() instead of matplotlib.pyplot.plot().

import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)
import seaborn as sns #sets up styles and gives us more plotting options
#import DataFrame as df
# Import all libraries needed for the tutorial

# General syntax to import specific functions in a library:
##from (library) import (specific library function)
from pandas import DataFrame, read_csv
from sklearn.cluster import KMeans
from classification import ex_d
# General syntax to import a library but no functions:
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number
import re




path= r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_cnt.txt'
data=pd.read_csv(path,header=None)
path2=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt'
mark=pd.read_csv(path2,header=None)




electrode_list=[32,33,34,36,37,38,42,43,44,45,46,47,50,51,52,54,55,56,59,60,61,62,63,64,68,69,70,72,73,74]
# converted data txt file in  2-d array
data_array=[]
for r in range(0,data.shape[0]-1):
    data_array.append(map(float,data.iloc[r,0].split('\t')))
#converted mark txt file in 2-d array
mark_array=[]
for d in range(0,mark.shape[0]-1):
    mark_array.append(map(float,mark.iloc[d,0].split('\t')))

array=[]
for r in range(1,119):
    array.append(r)
# 2-d array mark_array  converted in to data frame
marker_data=pd.DataFrame(mark_array,columns=['Time','Number'])
data_value=pd.DataFrame(data_array)
data_value.columns=array

print marker_data.shape[0]
x= marker_data.shape[0]
#if x%2!=0 :
mean1=np.empty((30,300,marker_data.shape[0]))
#mean2=np.empty((118,300,(marker_data.shape[0]+1)/2))
#else :
#mean1=np.empty((118,300,marker_data.shape[0]/2))
#mean2=np.empty((118,300,marker_data.shape[0]/2+1))
for r in range(marker_data.shape[0]-1):
    x= marker_data.iloc[r,0]
    i=-1
    for f in electrode_list:
        i+=1
        for d in range(int(x),int(x)+300):
            #if r%2!=0:conda install scikit-learnprint
            mean1[i,d-int(x),r]=data_value.iloc[d,f]
            #else:
            #    mean2[f,d-int(x),r/2-1]=data_value.iloc[d,f]
#dp = np.zeros((168,30))
#x= mean1[2:3,4:5,6:8]
#print x

#for i in range(30):
#    for j in range(300):
#            for k in range(168):
#                dp[k][i] += mean1[i][j][k]
kmean_object=ex_d(r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_cnt.txt',r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt',168,300,0,1)
data=kmean_object.wave_let()
#data is (7*168)*118
weights=[0.0,0.0,0.0,0.0,0.0,0.5,0.4,0.1]
kmean_data=np.zeros((168,944/118*30))
k=np.shape(data)
print k[0]
print k[1]
d=0
for i in range(0,k[0]):
    for r in range(0,k[1],8):
        if r in electrode_list:
            print r
            for j in range(0,8):
                kmean_data[i,d]+=weights[j]*data[i][r+j]
                d+=1
print kmean_data
km = KMeans(n_clusters=2)
km.fit(kmean_data)
labels = km.labels_
print(labels)
count=0
for i in range(0,k[0]):
    if marker_data.iloc[i,1]-labels[i]==1:
        count+=1
print "accuracy is"
print count/float(k[0])*100
#fs=300
#x=np.arange(fs)
#y=[mean1[0,i,0] for i in np.arange(fs)]
#z=[mean2[0,i,0] for i in np.arange(fs)]


#for k in range (0,30):
#    y=[mean1[k,i,47] for i in np.arange(fs)]
#    z=[mean1[k,i,42] for i in np.arange(fs)]
#    plt.figure()
#    plt.plot(x,y,label="1")
#    plt.plot(x,z,label="2")
#plt.show()
