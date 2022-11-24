from ast import For
from frechetdist import frdist
from sklearn import linear_model
import sklearn
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from matplotlib.backends.backend_pdf import PdfPages

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


from threading import Thread, Lock, RLock
from multiprocessing import Pool
import time
import math

from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

algo_config_options = [16, 19, 32, 11, 18, 54, 44, 39, 40, 42]
algo_steps = [3, 3, 3, 3, 3, 3, 10, 20, 10, 20]
ratioGeo = [2, 2, 2, 2, 2, 2, 4, 4, 4, 4]
maxSamplingDistance = [5, 5, 5, 5, 5, 5, 25, 50, 25, 50]
maxConfiguration = [1152,432,2304,1024,2560,13485,68640,193536,60000,216000]

algo = [ "lrzip", "LLVM","x264","Dune",  "BerkeleyDBC",
        "Hipacc", "Polly","7z", "JavaGC",  "VP9"]
numRuns = 100  # number of repetitions performed

min_impurity_percent = 0.01
critn = "friedman_mse"

# Geometryc
startGeo = 100
thresholdList = [500, 200  , 150, 100, 50, 20, 10, 1]

ratio = 1.2

def annotate_boxplot(bpdict, annotate_params=None,
                     x_offset=0.05, x_loc=0,
                     text_offset_x=35,
                     text_offset_y=20):
    """Annotates a matplotlib boxplot with labels marking various centile levels.

    Parameters:
    - bpdict: The dict returned from the matplotlib `boxplot` function. If you're using pandas you can
    get this dict by setting `return_type='dict'` when calling `df.boxplot()`.
    - annotate_params: Extra parameters for the plt.annotate function. The default setting uses standard arrows
    and offsets the text based on other parameters passed to the function
    - x_offset: The offset from the centre of the boxplot to place the heads of the arrows, in x axis
    units (normally just 0-n for n boxplots). Values between around -0.15 and 0.15 seem to work well
    - x_loc: The x axis location of the boxplot to annotate. Usually just the number of the boxplot, counting
    from the left and starting at zero.
    text_offset_x: The x offset from the arrow head location to place the associated text, in 'figure points' units
    text_offset_y: The y offset from the arrow head location to place the associated text, in 'figure points' units
    """
    if annotate_params is None:
        annotate_params = dict(xytext=(text_offset_x, text_offset_y), textcoords='offset points', arrowprops={'arrowstyle':'->'})

    plt.annotate('Median', (x_loc + 1 + x_offset, bpdict['medians'][x_loc].get_ydata()[0]), **annotate_params)
    plt.annotate('25%', (x_loc + 1 + x_offset, bpdict['boxes'][x_loc].get_ydata()[0]), **annotate_params)
    plt.annotate('75%', (x_loc + 1 + x_offset, bpdict['boxes'][x_loc].get_ydata()[2]), **annotate_params)
    plt.annotate('5%', (x_loc + 1 + x_offset, bpdict['caps'][x_loc*2].get_ydata()[0]), **annotate_params)
    plt.annotate('95%', (x_loc + 1 + x_offset, bpdict['caps'][(x_loc*2)+1].get_ydata()[0]), **annotate_params)


if __name__ == "__main__":

    numbersConfiguration = np.empty((0, 8), int)
    numbersPrediction = np.empty((0, 8), int)
    for id in range(0,10):
        
        file = "dataProcessedEstimated_"+str(algo[id])+".csv"
        print(file)
        colnames = ['error','min','q1','med','q3','max','minC','q1C','medC','q3C','maxC','minP','q1P','medP','q3P','maxP']
        
        dataSummary = pd.read_csv(file,names=colnames,header=None)
        #print(dataSummary)
        dataSummary.index=["5","2","1.5","1","0.5","0.2","0.1","0.01"]
        dataSummary=dataSummary.transpose()
      
        
        
        
        print("Configuration")
        
        mierda = np.array([dataSummary.iloc[3,:].to_numpy()])
        print(mierda)
        numbersConfiguration=np.append(numbersConfiguration,mierda,axis=0)   
     

        print("Prediction")
        mierda = np.array([dataSummary.iloc[13,:].to_numpy()])
        print(mierda.round(2))
        numbersPrediction= np.append(numbersPrediction,mierda,axis=0)   
        print("") 
    print("**************")    
    print('Number of configuration')    
    print(numbersConfiguration.round(2)) 
    print('Prediction')    
    print(numbersPrediction.round(2))          
 
        
            
          
        


        
        
