# Import modules
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math
import pickle

countingTimePoint = np.array([80,65,70])
quiescent == True

if quiescent == True:
    # Simulation for clonal growth using quiescent cell dynamics model
    parameters = {
        "k1":0.0225,
        "k2":0.1275,
        "k3":0.01,
        "lambda":1.0,
        "k6":0.01,
        "k7":0.48,
        "k8":0.01}
    from QuiescentCellDynamicsModel import passageClonalGrowth
else:
    # Simulation for clonal growth using stem cell hierarchy model
    parameters = {
        "stem_div_rate":0.15,
        "epsilon":0.15,
        "pro_div_rate":1.0,
        "apoptosis_rate":0.48}
    from StemCellHierarchyModel import passageClonalGrowth

# Measure the number of cells at each time point
def cellCount(reactionTime, cell, countingTimePoint):
    time = range(countingTimePoint)
    count = np.zeros(len(time))
    for t in time:
        cellIndex = np.array(np.where(reactionTime <= t)[0])
        if cellIndex.size:
            cellTimeIndex = np.array(cellIndex[-1])
            count[t] = cell[cellTimeIndex]
        else:
            count[t] = cell[0]
    return(count)

# decide the number of cells for each cell type to be transplanted
def transplantation(harvest,transplantationProb, quiescent):
    stemHarvest = harvest["stem"][-1]
    proHarvest = harvest["progenitor"][-1] 
    if quiescent == True:
    	quieHarvest = harvest["quiescent"][-1]
    	totalHarvest = stemHarvest + proHarvest + quieHarvest
    else:
    	totalHarvest = stemHarvest + proHarvest
#    only stem cells and progenitor cells may survive during the process of serial transplantation. 
    if totalHarvest > 0:
        stemProb = stemHarvest/totalHarvest
        proProb = proHarvest/totalHarvest
        stemReinject = np.random.binomial(n=totalHarvest, p=stemProb*transplantationProb)
        proReinject = np.random.binomial(n=totalHarvest, p=proProb*transplantationProb)
        if quiescent == True:
        	quieProb = quieHarvest/totalHarvest
        	quieReinject = np.random.binomial(n=totalHarvest, p=quieProb*transplantationProb)
        else:
        	pass
    else:
        stemReinject = 0
        proReinject = 0
        quieReinject = 0
    if quiescent == True:
    	return{"stem":np.array([stemReinject]), "progenitor":np.array([proReinject]), "quiescent":np.array([quieReinject])}
    else:
    	return{"stem":np.array([stemReinject]), "progenitor":np.array([proReinject]), "differentiated":np.array([0])}

# Simulate the clonal growth across serial transplantation
def passageExpansion(primaryInject, parameters, countingTimePoint, transplantationProb, quiescent):
    primaryHarvest = passageClonalGrowth(primaryInject, parameters,countingTimePoint[0])
    secondaryInject = transplantation(primaryHarvest, transplantationProb, quiescent)
    secondaryHarvest = passageClonalGrowth(secondaryInject, parameters,countingTimePoint[1])
    tertiaryInject = transplantation(secondaryHarvest, transplantationProb, quiescent)
    tertiaryHarvest = passageClonalGrowth(tertiaryInject, parameters,countingTimePoint[2])
    S = np.concatenate((primaryHarvest["stem"],secondaryHarvest["stem"],tertiaryHarvest["stem"]))
    P = np.concatenate((primaryHarvest["progenitor"],secondaryHarvest["progenitor"],tertiaryHarvest["progenitor"]))
    D = np.concatenate((primaryHarvest["differentiated"],secondaryHarvest["differentiated"],tertiaryHarvest["differentiated"]))
    if quiescent == True:
        Q = np.concatenate((primaryHarvest["quiescent"],secondaryHarvest["quiescent"],tertiaryHarvest["quiescent"]))
    else:
        Q = np.zeros(len(S))
    return{"stem":S, "progenitor":P, "differentiated":D, "quiescent":Q}

# Simulate the growth of multiple clones. The initial cell type for clonal growth is determined based of cell composition at steady status.
def multiGrowthSimulation(quiescent):
    # N: the number of parallele tasks   
    simulation = 10000
    if quiescent == True:
        s = np.random.binomial(n=simulation, p=0.17)
        p = np.random.binomial(n=simulation, p=0.64)
    else:
        s = np.random.binomial(n=simulation, p=stemPrimaryProb)
    timeRange = int(sum(countingTimePoint))
    if quiescent == True:
        multiGrowth = np.zeros((simulation,4,timeRange))
        for i in range(simulation):
            if i < s:
                clone = passageExpansion({"stem":1,"progenitor":0,"quiescent":0}, parameters, countingTimePoint, 0.37, quiescent)
            elif i < (s + p):
                clone = passageExpansion({"stem":0,"progenitor":1,"quiescent":0}, parameters, countingTimePoint, 0.37, quiescent)
            else:
                clone = passageExpansion({"stem":0,"progenitor":0,"quiescent":1}, parameters, countingTimePoint, 0.37, quiescent)
            multiGrowth[i][0][:] = clone["stem"]
            multiGrowth[i][1][:] = clone["progenitor"]
            multiGrowth[i][2][:] = clone["differentiated"]
            multiGrowth[i][3][:] = clone["quiescent"]
    else:
        multiGrowth = np.zeros((simulation,3,timeRange))
        for i in range(simulation):
            if i < s:
                clone = passageExpansion({"stem":1,"progenitor":0}, parameters, countingTimePoint, 0.37, quiescent)
            else:
                clone = passageExpansion({"stem":0,"progenitor":1}, parameters, countingTimePoint, 0.37, quiescent)
            multiGrowth[i][0][:] = clone["stem"]
            multiGrowth[i][1][:] = clone["progenitor"]
            multiGrowth[i][2][:] = clone["differentiated"]
    return(multiGrowth)

multiGrowth = multiGrowthSimulation(quiescent)
pickle_out = open('multiple_clonal_growth','wb')
pickle.dump(multiGrowth, pickle_out)
pickle_out.close()

# The average clonal growth is measured from the multiple clones
def meanGrowth(multiGrowth, countingTimePoint, quiescent):
    shape = np.shape(multiGrowth)
    stem = np.zeros((shape[0],shape[2]))
    pro = np.zeros((shape[0],shape[2]))
    diff = np.zeros((shape[0],shape[2]))
    quie = np.zeros((shape[0],shape[2]))
    for i in range(shape[0]):
        stem[i][:] = multiGrowth[i][0]
        pro[i][:] = multiGrowth[i][1]
        diff[i][:] = multiGrowth[i][2]
        if quiescent == True:
            quie[i][:] = multiGrowth[i][3]
        else:
            pass
    S = np.transpose(stem)
    P = np.transpose(pro)
    D = np.transpose(diff)
    Q = np.transpose(quie)
    meanS = np.array([np.mean(s) for s in S])
    meanP = np.array([np.mean(p) for p in P])
    meanD = np.array([np.mean(d) for d in D])
    meanQ = np.array([np.mean(q) for q in Q])
    return{"stem":meanS, "progenitor":meanP, "differentiated":meanD, "quiescent":meanQ}

averageGrowth = meanGrowth(multiGrowth, countingTimePoint, quiescent)

# Plot how the cell number for each cell type change across time.
def cellExpansionPlot(harvest, quiescent):
    S = harvest["stem"]
    P = harvest["progenitor"]
    D = harvest["differentiated"]
    x = range(len(S))
    plt.plot(x,S,label='stem cell')
    plt.plot(x,P,label='progenitor cell')
    plt.plot(x,D,label='differentiated cell')
    if quiescent == True:
        Q = harvest["quiescent"]
        plt.plot(x,Q,label='quiescent cell')
    else:
        pass
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('cell count')

cellExpansionPlot(averageGrowth)
plt.savefig('cell_level_average_clonal_growth.pdf')
plt.clf()

# Plot how the total number of cells (the clone size) change across time.
def cloneGrowthPlot(harvest, quiescent):
    S = harvest["stem"]
    P = harvest["progenitor"]
    D = harvest["differentiated"]
    if quiescent == True:
        Q = harvest["quiescent"]
    else:
        Q = np.zeros(len(S))
    total = np.array([sum((s,p,d,q)) for (s,p,d,q) in zip(S, P, D, Q)])
    x = range(len(S))
    plt.plot(x,total)
    plt.xlabel('time')
    plt.ylabel('clone size')

cloneGrowthPlot(averageGrowth, quiescent)
plt.savefig('average_clonal_growth.pdf')
plt.clf()

# Plot how the cell composition for each cell type change across time.
def checkSteadyPlot(harvest, quiescent):
    S = harvest["stem"]
    P = harvest["progenitor"]
    D = harvest["differentiated"]
    if quiescent == True:
        Q = harvest["quiescent"]
    else:
        Q = np.zeros (len(S))
    total = np.array([sum((s,p,d,q)) for (s,p,d,q) in zip(S, P, D, Q)])
    x = range(len(S))
    sProb = np.zeros(len(S))
    pProb = np.zeros(len(P))
    dProb = np.zeros(len(D))
    qProb = np.zeros(len(S))
    for i in x:
        if total[i] == 0:
            pass
        else:
            sProb[i] = S[i]/total[i]
            pProb[i] = P[i]/total[i]
            dProb[i] = D[i]/total[i]
            qProb[i] = Q[i]/total[i]
    plt.plot(x,sProb,label='stem cell')
    plt.plot(x,pProb,label='progenitor cell')
    plt.plot(x,dProb,label='differentiated cell')
    plt.plot(x,qProb,label='quiescent cell')
    plt.legend()
    plt.xlabel('time')
    plt.ylabel('cell composition')

checkSteadyPlot(averageGrowth)
plt.savefig('cell_composition.pdf')
plt.clf()

# Grap the clone size distribution at three harvest time points.
def cloneSizeDistribution(multiGrowth, countingTimePoint):
    clones = len(multiGrowth)
    primary = np.zeros(clones)
    secondary = np.zeros(clones)
    tertiary = np.zeros(clones)
    for clone in range(clones):
        primary[clone] = sum(np.array([multiGrowth[clone][cellType][int(countingTimePoint[0]-1)] for cellType in range(3)]))
        secondary[clone] = sum(np.array([multiGrowth[clone][cellType][int(sum(countingTimePoint[:2])-1)] for cellType in range(3)]))
        tertiary[clone] = sum(np.array([multiGrowth[clone][cellType][int(sum(countingTimePoint[:3])-1)] for cellType in range(3)]))
    return{"primary":primary, "secondary":secondary, "tertiary": tertiary}

cloneSize = cloneSizeDistribution(multiGrowth, countingTimePoint)

# The function to plot negative binomial distribution based on the parameters in the stem cell hierarchy model.
def negativeBinomialDistribution(divisionRate, countingTimePoint, n):
    n0 = divisionRate * countingTimePoint / 2
    N0 = np.log(n0)
    P = 1 / N0 * np.exp(-n/n0) / n
    return(P)

# Plot the clone size distribution in the logarithm scale.
def logSizeDistribution(cloneSizes,NBD,title):
    minimum = int(min(cloneSizes))
    print('minimum clone size = ' + str(minimum))
    maximum = int(max(cloneSizes))
    print('maximum clone size = ' + str(maximum))
    x = np.asarray(range(maximum))
    sizes = list(np.array(cloneSizes))
    freq = np.asarray([sizes.count(i+1) for i in range(maximum)])
    plt.bar(x,freq,width=2,color='y')
    y = NBD*len(cloneSizes)
    plt.plot(x, y, 'r')
    plt.xlabel('clone size')
    plt.ylabel('frequency')
    plt.title(title)

# Plot the first incomplete moment based on the clone size distribution at harvest
def firstIncompleteMoment(cloneSizes):
    maximum = int(max(cloneSizes))
    cloneSizeRange = np.asarray(range(maximum))
    sizes = list(np.array(cloneSizes))
    freq = np.asarray([sizes.count(i) for i in range(maximum+1)])
    prob = freq/sum(freq)
    averageCloneSize = np.mean(cloneSizes)
    step1 = np.array([i*p for (i,p) in enumerate(prob)])
    step4 = np.zeros(len(step1))
    for i in range(len(step1)):
        step4[i]= sum(step1[i:])
    mu = step4/averageCloneSize
    return(mu)

# Plot the first incomplete moment of the clone sizes at three harvests
def firstIncompleteMomentLogScalePlot(cloneSize):
    primary = cloneSize["primary"]
    primaryRange = np.asarray(range(int(max(primary)+1)))
    primaryMu = firstIncompleteMoment(primary)
    plt.plot(primaryRange, primaryMu, label = "primary")
    
    secondary = cloneSize["secondary"]
    secondaryRange = np.asarray(range(int(max(secondary)+1)))
    secondaryMu = firstIncompleteMoment(secondary)
    plt.plot(secondaryRange, secondaryMu, label = "secondary")
    
    tertiary = cloneSize["tertiary"]
    tertiaryRange = np.asarray(range(int(max(tertiary)+1)))
    tertiaryMu = firstIncompleteMoment(tertiary)
    plt.plot(tertiaryRange, tertiaryMu, label = "tertiary")
    
    plt.legend()
    plt.xlabel('clone size')
    plt.ylabel('log first incomplete moment')
    plt.yscale('log')

firstIncompleteMomentLogScalePlot(cloneSize)
plt.savefig('first_incomplete_moment.pdf')
plt.clf()

