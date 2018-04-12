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
from sklearn.neighbors.nearest_centroid import NearestCentroid


path=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/clustering_data.txt'
data=np.loadtxt(path)
data=np.array(data)
path2=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/target_data.txt'
targets=np.loadtxt(path2)
targets=np.array(targets)


array=np.array(data)



kf = KFold(n_splits=10,random_state = 30, shuffle = True)     #splitting into 10
kf.get_n_splits(array)
for train_index, test_index in kf.split(array):
    X_train, X_test = array[train_index],array[test_index]
    y_train, y_test = targets[train_index], targets[test_index]




    experiment=svm.SVC(kernel='linear')
    experiment.fit(X_test,y_test)
    targets=np.array(targets)
    count=0
    total=len(y_test)

    predict=[]
    predict=experiment.predict(X_test)
    for x in range(len(predict)) :
        if y_test[x]==predict[x] :
            count+=1
    #print predict        #count+=1
    print "svm_accuracy with linear kernel", float(count)/len(predict)*100

    experiment=svm.SVC(kernel='rbf')
    experiment.fit(X_test,y_test)
    targets=np.array(targets)
    count=0
    total=len(y_test)
    #for x,y  in zip(X_train,y_train):
    predict=[]
    predict=experiment.predict(X_test)
    for x in range(len(predict)) :
        if y_test[x]==predict[x] :
            count+=1
    #print predict        #count+=1
    print "svm_accuracy with rbf kernel", float(count)/len(predict)*100


    experiment=svm.SVC(kernel='sigmoid')
    experiment.fit(X_test,y_test)
    targets=np.array(targets)
    count=0
    total=len(y_test)
    #for x,y  in zip(X_train,y_train):
    predict=[]
    predict=experiment.predict(X_test)
    for x in range(len(predict)) :
        if y_test[x]==predict[x] :
            count+=1
    #print predict        #count+=1
    print "svm_accuracy with sigmoid kernel", float(count)/len(predict)*100


    experiment=svm.SVC(kernel='poly')
    experiment.fit(X_test,y_test)
    targets=np.array(targets)
    count=0
    total=len(y_test)
    #for x,y  in zip(X_train,y_train):
    predict=[]
    predict=experiment.predict(X_test)
    for x in range(len(predict)) :
        if y_test[x]==predict[x] :
            count+=1
    #print predict        #count+=1
    print "svm_accuracy with poly kernel", float(count)/len(predict)*100
    knn = NearestCentroid()
    knn.fit(X_test, y_test)
    count=0
    total=len(y_test)
    #for x,y  in zip(X_train,y_train):
    predict=[]
    predict=knn.predict(X_test)
    for x in range(len(predict)) :
        if y_test[x]==predict[x] :
            count+=1
    print "knn0",float(count)/len(predict)*100

    count=0
    from sklearn import tree
    decisiontree = tree.DecisionTreeClassifier()
    decisiontree = decisiontree.fit(X_train, y_train)
    predict=decisiontree.predict(X_test)
    for x in range(len(predict)) :
        if y_test[x]==predict[x] :
            count+=1
    #print predict        #count+=1
    print "Decision_tree", float(count)/len(predict)*100

#kernel = C(1.0, (1e-3, 1e3)) * RBF([5,5], (1e-2, 1e2))

from sklearn import datasets
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel
iris = datasets.load_iris()
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
array,targets=iris.data,iris.target
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

'''iris = datasets.load_iris()

gnb = GaussianNB(kernel='RBF')
array,targets=iris.data,iris.target
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

iris = datasets.load_iris()
gnb = GaussianNB(kernel='RationalQuadratic')
array,targets=iris.data,iris.target
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

iris = datasets.load_iris()
gnb = GaussianNB(kernel='ExpSineSquared')
array,targets=iris.data,iris.target
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

iris = datasets.load_iris()
gnb = GaussianNB(kernel='Matern')
array,targets=iris.data,iris.target
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

iris = datasets.load_iris()
gnb = GaussianNB(kernel='DotProduct')
array,targets=iris.data,iris.target
y_pred = gnb.fit(iris.data, iris.target).predict(iris.data)
print("Number of mislabeled points out of a total %d points : %d"
      % (iris.data.shape[0],(iris.target != y_pred).sum()))

'''
'''count=0
km = KMeans(n_clusters=2,max_iter = 5000,n_init = 150)
km.fit(array)
labels = km.labels_
print labels
for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
    #print predict        #count+=1

print "Kmeans:", float(count)/len(labels)*100

count=0
mkm = MiniBatchKMeans(n_clusters=2,max_iter = 5000,n_init = 50)
mkm.fit(array)
labels = mkm.labels_
for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
#print predict        #count+=1

print "MIniBatchKMeans:", float(count)/len(labels)*100

count=0
mkm = SpectralClustering(n_clusters=2)
mkm.fit(array)
labels = mkm.labels_
for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
#print predict        #count+=1

print "SpectralClustering:", float(count)/len(labels)*100

count=0
mkm = MeanShift()
mkm.fit(array)
labels = mkm.labels_
print(labels)
for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
#print predict        #count+=1

print "MeanShift:", float(count)/len(labels)*100
count=0
mkm = AffinityPropagation()
mkm.fit(array)
labels = mkm.labels_
for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
    #print predict        #count+=1

print "AffinityPropagation:", float(count)/len(labels)*100

count=0
mkm = AgglomerativeClustering(n_clusters =2)
mkm.fit(array)
labels = mkm.labels_
for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
    #print predict        #count+=1

print "AgglomerativeClustering:", float(count)/len(labels)*100
count=0
mkm = DBSCAN()
mkm.fit(array)
labels = mkm.labels_

for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
    #print predict        #count+=1

print "DBSCAN:", float(count)/len(labels)*100



count=0
mkm = Birch()
mkm.fit(array)
labels = mkm.labels_
for x in range(len(labels)) :
    if targets[x]==labels[x] :
        count+=1
    #print predict        #count+=1

print "Birch:", float(count)/len(labels)*100
'''
