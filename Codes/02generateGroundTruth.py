#!/usr/bin/env python3
from sklearn import linear_model
import sklearn
import pandas as pd
import scipy.stats as stats
import numpy as np
from sklearn.tree import DecisionTreeRegressor

from sklearn import metrics
from sklearn.metrics import mean_absolute_percentage_error


from threading import Thread, Lock, RLock
from multiprocessing import Pool
import time


algo_config_options = [16, 19, 32, 11, 18, 54, 44, 39, 40, 42]
algo_steps = [1, 1, 1, 1, 1, 1, 7, 10, 6, 22]
algo = ["x264", "lrzip", "Dune", "LLVM", "BerkeleyDBC",
        "Hipacc", "7z", "JavaGC", "Polly", "VP9"]
numRuns = 100  # number of repetitions performed

min_impurity_percent = 0.01
critn = "friedman_mse"


def unTest(dataTotalRandom, id):
    dataTotalRandom = dataTotalRandom
    id = id
    scoreGeneralRandomRow = np.zeros(
        (round(len(dataTotalRandom.index)/algo_steps[id]+algo_steps[id]-5), 1))

    m_depth = int(round(np.sqrt(algo_config_options[id])))

    min_impurity = float((dataTotalRandom[["performance"]].max(
        axis=0) - dataTotalRandom[["performance"]].min(axis=0))*min_impurity_percent)

    lenTotal = round(len(dataTotalRandom.index) /
                     algo_steps[id]+algo_steps[id]-5)

    dataTotalRandom = dataTotalRandom.sample(
        frac=1).reset_index(drop=True)
    XTotalRandom = dataTotalRandom.loc[:,
                                       dataTotalRandom.columns != 'performance']
    yTotalRandom = np.transpose(
        dataTotalRandom.loc[:, dataTotalRandom.columns == 'performance'].values)
    yTotalRandom = yTotalRandom[0]

    for nti in range(0, lenTotal, 1):

        nt = 6+nti*algo_steps[id]
        X = XTotalRandom[1:nt].copy()
        y = yTotalRandom[1:nt].copy()

        dt = DecisionTreeRegressor(
            random_state=0, criterion=critn, max_depth=m_depth, min_impurity_decrease=min_impurity)

        model = dt.fit(X, y)

        errora = mean_absolute_percentage_error(
            yTotalRandom, model.predict(XTotalRandom))
        scoreGeneralRandomRow[nti] = errora

    return scoreGeneralRandomRow


if __name__ == "__main__":

    for id in range(0, 12):
        start = time.time()
        file = "dataProcessed"+str(algo[id])+".csv"

        dataTotalRandom = pd.read_csv(file)
        # print(dataTotalRandom.head())

        print(str(algo[id])+" - MaxPerformance " + str(max(dataTotalRandom.loc[:, dataTotalRandom.columns == 'performance'])
                                                       ) + " - MinPerformance " + str(min(dataTotalRandom.loc[:, dataTotalRandom.columns == 'performance'])))
        print(str(algo[id])+" init 6 step " + str(algo_steps[id]) +
              " end " + str(len(dataTotalRandom.index)))
        # Algoritmo aleatorio a saco paco.
        scoreGeneralRandom = np.zeros(
            (round(len(dataTotalRandom.index)/algo_steps[id]+algo_steps[id]-5), numRuns))
        scoreGeneralRandomMeanSd = np.zeros(
            (round(len(dataTotalRandom.index)/algo_steps[id]+algo_steps[id]-5), 2))

        runsArray = []  # empty array
        # Step 1: Init multiprocessing.Pool()
        with Pool(processes=None) as pool:         # start 4 worker processes
            # evaluate "f(10)" asynchronously in a single process
            for run in range(0, numRuns):
                runsArray.append(pool.apply_async(
                    unTest, (dataTotalRandom, id)))

            for run in range(0, numRuns):
                runsArray[run].wait()

            for run in range(0, numRuns):
                scoreGeneralRandom[:, run] = runsArray[run].get().ravel()

        scoreGeneralRandomMeanSd[:, 0] = np.mean(scoreGeneralRandom, axis=(1))
        scoreGeneralRandomMeanSd[:, 1] = np.std(scoreGeneralRandom, axis=(1))

        nameFile = "dataProcessedGroundTruthTreeNEGMEAN" + \
            str(algo[id]) + ".csv"
        print("dataProcessedGroundTruthTreeNEGMEAN" + str(algo[id]) + ".csv")

        pd.DataFrame(scoreGeneralRandomMeanSd).to_csv(
            nameFile, index=False, header=False)
        end = time.time()
        print("The time of execution of above program is :", end-start)
