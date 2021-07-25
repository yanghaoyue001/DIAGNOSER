# %matplotlib inline
import numpy as np # imports a fast numerical programming library
import scipy as sp #imports stats functions, amongst other things
import matplotlib as mpl # this actually imports matplotlib
import matplotlib.cm as cm #allows us easy access to colormaps
import matplotlib.pyplot as plt #sets up plotting under plt
import pandas as pd #lets us handle data as dataframes
import seaborn as sns #sets up styles and gives us more plotting options
from sklearn import tree
from sklearn import ensemble
from sklearn.externals.six import StringIO

### for x data: 1,2,3 --> 1; 4,5,6 --> 0; 7,8,9 --> -1

#sets up pandas table display
pd.set_option('display.width', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.notebook_repr_html', True)

# input dataset
dataset_file_path = "./MAG COVID EVAL/mag-05312021/demo-images/training data 1.csv"
dataFrame = pd.read_csv(dataset_file_path, sep=',',  header=None, engine='python')
dataFrame = dataFrame.astype(int)
dataFrame.head()

all_Data=dataFrame.to_numpy()
np.random.shuffle(all_Data)

X_Train=all_Data[0:round(all_Data.shape[0]*1),0:all_Data.shape[1]-1]
Y_Train=all_Data[0:round(all_Data.shape[0]*1),all_Data.shape[1]-1]

X_Test=all_Data[round(all_Data.shape[0]*0.3):,0:all_Data.shape[1]-1]
Y_Test=all_Data[round(all_Data.shape[0]*0.3):,all_Data.shape[1]-1]


