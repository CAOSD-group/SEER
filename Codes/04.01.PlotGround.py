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

import statistics as st

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
#x264", "lrzip", "Dune", "LLVM", "BerkeleyDBC","Hipacc", "7z", "JavaGC", "Polly", "VP9"
t0 = [12,18,25,11,15,49,39,32,28,31]
t1 = [65,93,265,55,97,727,600,468,345,483]
t2 = [212,181,1071,165,363,4131,4091,3504,2172,3893]


minCon = [15,15,15,15,15,15,3,3,15]

algo = ["x264", "lrzip", "Dune", "LLVM", "BerkeleyDBC",
        "Hipacc", "7z", "JavaGC", "Polly", "VP9"]
numRuns = 100  # number of repetitions performed

min_impurity_percent = 0.01
critn = "friedman_mse"

# Geometryc
startGeo = 100
thresholdList = [5, 2  , 1.5, 1.0, 0.5, 0.2, 0.1, 0.01, 0]






ratio = 1.2

def inversepowerlaw(x, a,b,c): return a+b*(x**(-c))



if __name__ == "__main__":

    with PdfPages('salidaGTvsEstimated0.pdf') as pdf:

        for id in range(0,10):
            
            fileGround = "dataProcessedGroundTruthTreeNEGMEAN"+str(algo[id])+"MedidaError.csv" 
            
            file = "dataProcessedEstimated_"+str(algo[id])+".csv"

            colnames = ['error','min','q1','med','q3','max','minC','q1C','medC','q3C','maxC','minP','q1P','medP','q3P','maxP']
            
            dataProcessed = pd.read_csv(file,names=colnames,header=None)
            dataProcessed.index=["5","2","1.5","1","0.5","0.2","0.1","0.01"]
            dataProcessed=dataProcessed.transpose()
            
            
            
            filePopt0 = "dataProcessedEstimatedPopt_" + str(algo[id]) + "_0.csv"
            
            dataSummary = pd.read_csv(fileGround,header='infer')

            
            dataPopt0 = pd.read_csv(filePopt0,header=None)
            dataPopt0.columns=['a','b','c']

            

            dataFitX = dataSummary["numeroconfiguraciones"] 
  

            #plt.title(str(algo[id]))
            plt.xlabel("Sampling size")
            plt.ylabel("Relative Error")
            sizes =  dataFitX
            scores_mean = dataSummary["mediaerror"]
    
            #print(random_scores_mean)
            #print(scoreGeneralRandom)
            scores_std = dataSummary["deviacion"]

            print(algo[id])
            print(str(round(100*(dataProcessed.iloc[8,0]/np.max(dataFitX)),2))+"%")
            print(dataProcessed.iloc[8,0])
      


            # Plot learning curve
            plt.grid()
  
    
            estimatedValuesNew = inversepowerlaw(dataFitX, *np.quantile(dataPopt0,0.5,axis=(0)))/100
            plt.plot(
                dataFitX, estimatedValuesNew, "-", color="b"
            )
            #plt.text(5, np.max(estimatedValuesNew), str(np.min(estimatedValuesNew)), ha='center', va='center', fontsize=16, color='blue',backgroundcolor='white')
           
            plt.fill_between(
                sizes,
                scores_mean - scores_std,
                scores_mean + scores_std,
                alpha=0.1,
                color="r"
            )
            plt.plot(
                sizes, scores_mean, "-", color="r",label="Ground Truth"
            )
            #plt.text(10, 0.1, str(np.min(scores_mean)), ha='center', va='center', fontsize=16, color='red',backgroundcolor='white')
           
          

            plt.xscale('log')
            #plt.ylim([0,5])
       
 
   
            pdf.savefig( bbox_inches='tight' )
            plt.close()
        
        


        
        
