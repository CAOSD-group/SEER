import os
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import train_test_split

from multiprocessing import Pool
import time
import math

from scipy.optimize import curve_fit

algo_config_options = [16, 19, 32, 11, 18, 54, 44, 39, 40, 42]
algo_steps = [3, 3, 3, 3, 3, 3, 10, 20, 10, 20]
ratioGeo = [2, 2, 2, 2, 2, 2, 4, 4, 4, 4]
maxSamplingDistance = [5, 5, 5, 5, 5, 5, 25, 50, 25, 50]

algo = ["x264", "lrzip", "Dune", "LLVM", "BerkeleyDBC",
        "Hipacc", "7z", "JavaGC", "Polly", "VP9"]
numRuns = 10  # number of repetitions performed

stdFactor = 0

min_impurity_percent = 0.01
critn = "friedman_mse"

# Geometryc
startGeo = 100
#thresholdList = [1 , 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
thresholdList = [500, 200  , 150, 100, 50, 20, 10,1]

algo_config_options = [16, 19, 32, 11, 18, 54, 44, 39, 40, 42]

def getProbability(dataGround,contErrorThreshold,numConfig):
    clThres = dataGround.iloc[:,[0,contErrorThreshold+3]]

    quy= "numeroconfiguraciones>="+str(numConfig)
    
    res=clThres.query(quy)
    if res.empty:
        res = clThres.iloc[-1,1]
    else:
        res=clThres.query(quy).iloc[0,1]
    return res
            
#1/(beta * (x**gamma))
def inversepowerlaw(x, a,b,c): return a+b*(x**(-c))


def unTest(dataTotalRandom, id):

    m_depth = int(round(np.sqrt(algo_config_options[id])))

    min_impurity = float((dataTotalRandom[["performance"]].max(
        axis=0) - dataTotalRandom[["performance"]].min(axis=0))*min_impurity_percent)

    dataTotalRandom = dataTotalRandom.sample(
        frac=1).reset_index(drop=True)
    XTotalRandom = dataTotalRandom.loc[:,
                                       dataTotalRandom.columns != 'performance']
    yTotalRandom = np.transpose(
        dataTotalRandom.loc[:, dataTotalRandom.columns == 'performance'].values)
    yTotalRandom = yTotalRandom[0]

    nti = 20
    ntiOld=10
    finish = False
    finishInterFail = False
    finishOld = False
    ini = False
    
    yFret = []
    dff=0
    while (not finish):
#ese log es para que llegemos en la progesion como maximo al valor de nti
  #      progressionAux = [round(10*ratio**i) for i in range(0, nti)]
        #i=0
        # ratio = 1+np.sqrt(algo_config_options[i])/10
        # x=round(10*ratio**i)
        # progression=[]
        # while (x < nti):
        #     progression.append(x)
        #     i+=1
        #     x=round(10*ratio**i)
        i=0
        x=10
        progression=[]
        while (x < nti):
            progression.append(x)
            i+=1
            x=int(np.round(x+np.sqrt(algo_config_options[id])))
          
        progression.append(nti)
        dataCurve = np.zeros((len(progression), numRuns))
        cont = 0
        for run in range(0, numRuns):
            cont = 0
            for nt in progression:
                X_train, X_test, y_train, y_test = train_test_split(
                    XTotalRandom[1:nti].copy(), yTotalRandom[1:nti].copy(), test_size=0.3)
                    
                X_train = X_train[1:nt];y_train = y_train[1:nt]
                X_test = X_test[1:nt]
                y_test = y_test[1:nt]
               

                dt = DecisionTreeRegressor(
                    random_state=0, criterion=critn, max_depth=m_depth, min_impurity_decrease=min_impurity)

                model = dt.fit(X_train, y_train)

                errora = mean_absolute_percentage_error(
                    y_test, model.predict(X_test))*100 #It is scale to percent to have higer number and decrease the error during fitting
                dataCurve[cont, run] = errora
                cont = cont+1
    
     
        #dataFitY = np.median(dataCurve, axis=(1))
        #dataFitY = np.median(dataCurve, axis=(1))
        #dataFitY = np.add(dataFitY,stdFactor*np.std(dataCurve, axis=(1)))
        dataFitY=np.quantile(dataCurve,1,axis=(1))
        
  
        dataEnd = dataCurve[cont-1,:]
        #

        belowYY=np.min(np.quantile(dataEnd,0.75,axis=(0)))

        coefficientVariation = np.std(dataEnd)/np.mean(dataEnd)
        

        aboveYY=np.min(np.quantile(dataEnd,0.25,axis=(0))-3*(np.quantile(dataEnd,0.75,axis=(0))-np.quantile(dataEnd,0.25,axis=(0))))
        if (aboveYY<0):
            aboveYY=0

      
        dataFitX = progression
        
   
        
        try:

            ntiOld = nti
            
            popt=[]

            popt, pcov = curve_fit(inversepowerlaw, dataFitX, dataFitY,bounds=((aboveYY,0,0),(belowYY,np.inf,5)))
   

            #estimatedValuesNew = inversepowerlaw(
            #    range(10, min(len(dataTotalRandom.index),1000), algo_steps[id]), *popt)
                
            if (not ini):
                ini=True
                #yFret = list(range(10, min(len(dataTotalRandom.index),1000), algo_steps[id]))
                #d1 = np.stack((estimatedValuesNew, yFret), axis=-1)
                d1 = popt[0]
         
            else:               
            
                
                #d2 = np.stack((estimatedValuesNew, yFret), axis=-1)   
                d2 = popt[0]  
                
                #dff = frdist(d1,d2)  
                dff=abs(d2-d1) 
                d1 = d2   
                finishOld = (dff < 10) or nti >= len(dataTotalRandom.index)       
                #finishOld =  nti >= len(dataTotalRandom.index)  
                finishInterFail = (popt[0]>aboveYY and popt[0]<= belowYY and popt[1]>0 and popt[2]>0 and popt[2]<=5) 
                finishSlope = coefficientVariation < 0.2
                finish = finishOld and finishInterFail and finishSlope
                #if (finishInterFail):
                #    print(dataFitX) 
                #    print(dataFitY)                
                nti += algo_steps[id]           

        except:
            nti += algo_steps[id]
            finish = finish or (nti >= len(dataTotalRandom.index))
       # finally:
           # if (finish):
             #   print("****FIN!"+ str(id) + "nti " + str(nti) + " " + str(round(max(algo_steps[id], nti*0.05))) + " ster" + str(dff) + str(ini) + " " + str(finish) + " " + str(finishInterFail) + " " + str(finishOld) + "  " + str(popt) + "  " + str(belowYY) + " CV " + str(coefficientVariation) + " encima " + str(aboveYY))      
           # else:
             #   print("ID"+ str(id) + "nti " + str(nti) + " " + str(round(max(algo_steps[id], nti*0.05))) + " ster" + str(dff) + str(ini) + " " + str(finish) + " " + str(finishInterFail) + " " + str(finishOld) + "  " + str(popt) + "  " + str(belowYY) + "Slope " + str(coefficientVariation) + " encima " + str(aboveYY))      

    
    return [ntiOld,popt]


if __name__ == "__main__":

    for id in range(0,10):

        scoreGeneralRandom = np.zeros((len(thresholdList), 16))
        scorePopt = pd.DataFrame([])
         
        scoreGeneralLocal = np.zeros((numRuns,len(thresholdList)))
        
        
        

        start = time.time()

        cur_path = os.path.dirname(__file__)

        file = os.path.join(cur_path,"../DataSet/dataProcessed"+str(algo[id])+".csv")

        dataTotalRandom = pd.read_csv(file)
        
        
        
        fileGround = os.path.join(cur_path,"../GroundTruth/dataProcessedGroundTruthTreeNEGMEAN"+str(algo[id])+"MedidaError.csv")
        dataGround= pd.read_csv(fileGround)
        

        
        print(file)

        print(str(algo[id])+" - MaxPerformance " + str(max(dataTotalRandom.loc[:, dataTotalRandom.columns == 'performance'])
                                                       ) + " - MinPerformance " + str(min(dataTotalRandom.loc[:, dataTotalRandom.columns == 'performance'])))
        print(str(algo[id])+" init 6 step " + str(algo_steps[id]) +
              " end " + str(len(dataTotalRandom.index)))

        # Maximum i should be the len/2. I.e. leng/2=startGeo*ratioGeo^i, log in both sides, and i equal to:
        nMaxSampleGeo = round(math.log(len(dataTotalRandom.index) /

                                       (2*startGeo))/math.log(ratioGeo[id]))

        sampleOption = [startGeo * ratioGeo[id] **
                        i for i in range(nMaxSampleGeo)]

        runsArray = []  # empty array
        estimateArrayNti = []  # empty array
        # Step 1: Init multiprocessing.Pool()
        with Pool(processes=None) as pool:
            # Progresive sampling
            cont = 0
            for run in range(0, numRuns):
                dataTotalRandom = dataTotalRandom.sample(
                    frac=1).reset_index(drop=True)
                dataTotalRandomF = dataTotalRandom.copy(deep=True)
                runsArray.append(pool.apply_async(
                    unTest, (dataTotalRandomF, id)))

            for run in range(0, numRuns):
                runsArray[run].wait()

            for run in range(0, numRuns):
               # print(type(runsArray[run].get()[1]))
                newRow=pd.DataFrame(runsArray[run].get()[1]).transpose()
               # print(newRow)
                scorePopt=pd.concat([scorePopt,newRow])
                
                estimateArrayNti.append(runsArray[run].get()[0])

                rangeI=range(6, len(dataTotalRandom.index))
                estimatedValuesNew = inversepowerlaw(
                        rangeI, *(runsArray[run].get()[1]))
                        
                #plt.scatter(range(6, 500, algo_steps[id]),inversepowerlaw(
                #        range(6, 500, algo_steps[id]), *(runsArray[run].get()[1])))
                #plt.show()
                for th in range(0, len(thresholdList)): 
                   
                    aux = next((x[0] for x in enumerate(estimatedValuesNew)
                            if x[1] <= thresholdList[th]), len(dataTotalRandom.index))+6
                    #There are some configurable system that   
                    scoreGeneralLocal[run,th]=aux

            #       
  
        nameFile = "dataProcessedEstimatedPopt_" + \
            str(algo[id]) + "_" + str(stdFactor)  + ".csv"

        pd.DataFrame(scorePopt).to_csv(
            nameFile, index=False, header=False)
            
            
        minValues =   np.min(scoreGeneralLocal,axis=0)
        firstValues =   np.quantile(scoreGeneralLocal,0.25,axis=0)
        meanValues =   np.quantile(scoreGeneralLocal,0.5,axis=0)
        thirdValues =   np.quantile(scoreGeneralLocal,0.75,axis=0)
        maxValues =   np.max(scoreGeneralLocal,axis=0)
        
        
        
     
        for contErrorThreshold in range(0, len(thresholdList)): 
            scoreGeneralRandom[contErrorThreshold, 0] = thresholdList[contErrorThreshold]
            scoreGeneralRandom[contErrorThreshold,1] = minValues[contErrorThreshold]
            scoreGeneralRandom[contErrorThreshold,2] = firstValues[contErrorThreshold]
            scoreGeneralRandom[contErrorThreshold,3] = meanValues[contErrorThreshold]
            scoreGeneralRandom[contErrorThreshold,4] = thirdValues[contErrorThreshold]
            scoreGeneralRandom[contErrorThreshold,5] = maxValues[contErrorThreshold]
            
            scoreGeneralRandom[contErrorThreshold,6] = np.min(estimateArrayNti)
            scoreGeneralRandom[contErrorThreshold,7] = np.quantile(estimateArrayNti,0.25)
            scoreGeneralRandom[contErrorThreshold,8] = np.quantile(estimateArrayNti,0.5)
            scoreGeneralRandom[contErrorThreshold,9] = np.quantile(estimateArrayNti,0.75)
            scoreGeneralRandom[contErrorThreshold,10] = np.max(estimateArrayNti)
            
        
            scoreGeneralRandom[contErrorThreshold,11] = getProbability(dataGround,contErrorThreshold,scoreGeneralRandom[contErrorThreshold,1])
            scoreGeneralRandom[contErrorThreshold,12] = getProbability(dataGround,contErrorThreshold,scoreGeneralRandom[contErrorThreshold,2])
            scoreGeneralRandom[contErrorThreshold,13] = getProbability(dataGround,contErrorThreshold,scoreGeneralRandom[contErrorThreshold,3])
            scoreGeneralRandom[contErrorThreshold,14] = getProbability(dataGround,contErrorThreshold,scoreGeneralRandom[contErrorThreshold,4])
            scoreGeneralRandom[contErrorThreshold,15] = getProbability(dataGround,contErrorThreshold,scoreGeneralRandom[contErrorThreshold,5])
            
            
            
  
    
        
        
        

        
        

        nameFile = "dataProcessedEstimated_" + \
            str(algo[id]) + ".csv"

        pd.DataFrame(scoreGeneralRandom).to_csv(
            nameFile, index=False, header=False)
        end = time.time()
        print("The time of execution of above program is :", end-start)

        print(nameFile)
