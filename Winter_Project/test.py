


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

# General syntax to import a library but no functions:
##import (library) as (give the library a nickname/alias)
import matplotlib.pyplot as plt
import pandas as pd #this is how I usually import pandas
import sys #only needed to determine Python version number
import matplotlib #only needed to determine Matplotlib version number

path2=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/data_set_IVa_aa_mrk.txt'
mark=pd.read_csv(path2,header=None)


mark_array=[]
for d in range(0,mark.shape[0]-1):
    mark_array.append(map(float,mark.iloc[d,0].split('\t')))

print mark_array[2][1]


'''path=r'/home/aakash/Desktop/MI_2_class_data/Training_data/aa/mean.txt'
mean_data=pd.read_csv(path,header=None)
#print mean_data
mean_array=[]
for r in range(0,mean_data.shape[0]-1):
    mean_array.append(map(float,mean_data.iloc[r,0].split(' ')))
#print mean_array
fs=277/2
x=np.arange(fs)
y=[mean_array[2*i+1][0] for i in np.arange(fs)]
z=[mean_array[2*i][0] for i in np.arange(fs)]
plt.plot(x,y,label="1")
plt.plot(x,z,label="0")
plt.legend()
plt.show()'''

'''
def fractal_dimension(Z, threshold=0.9):

    # Only for 2d image
    assert(len(Z.shape) == 2)

    # From https://github.com/rougier/numpy-100 (#87)
    def boxcount(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)

        # We count non-empty (0) and non-full boxes (k*k)
        return len(np.where((S > 0) & (S < k*k))[0])


    # Transform Z into a binary array
    Z = (Z < threshold)

    # Minimal dimension of image
    p = min(Z.shape)

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p)/np.log(2))

    # Extract the exponent
    n = int(np.log(n)/np.log(2))

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(Z, size))

    # Fit the successive log(sizes) with log (counts)
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
    return -coeffs[0]
'''
