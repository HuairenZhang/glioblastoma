import random
import numpy as np

def transplantation(harvest,transplantationProb):
    stemHarvest = harvest["stem"]
    proHarvest = harvest["progenitor"]
    quieHarvest = harvest["quiescent"]
    totalHarvest = stemHarvest + proHarvest + quieHarvest
    stemReinject = 0
    proReinject = 0
    quieReinject = 0
    if totalHarvest > 0:
        stemProb = stemHarvest/totalHarvest
        proProb = proHarvest/totalHarvest
        quieProb = quieHarvest/totalHarvest
        stemReinject += np.random.binomial(n=totalHarvest, p=stemProb*transplantationProb)
        proReinject += np.random.binomial(n=totalHarvest, p=proProb*transplantationProb)
        quieReinject += np.random.binomial(n=totalHarvest, p=quieProb*transplantationProb)
    else:
        pass
    return{"stem":stemReinject, "progenitor":proReinject, "quiescent":quieReinject}

def passageExpansion(primaryInject, testParameters, transplantationProb):
    primaryHarvest = clonalGrowth(primaryInject, testParameters,countingTimePoints[0])
    secondaryInject = transplantation(primaryHarvest,transplantationProb)
    secondaryHarvest = clonalGrowth(secondaryInject, testParameters,countingTimePoints[1])
    tertiaryInject = transplantation(secondaryHarvest,transplantationProb)
    tertiaryHarvest = clonalGrowth(tertiaryInject, testParameters,countingTimePoints[2])
    S = np.array([primaryHarvest["stem"],secondaryHarvest["stem"],tertiaryHarvest["stem"]])
    P = np.array([primaryHarvest["progenitor"],secondaryHarvest["progenitor"],tertiaryHarvest["progenitor"]])
    D = np.array([primaryHarvest["differentiated"],secondaryHarvest["differentiated"],tertiaryHarvest["differentiated"]])
    Q = np.array([primaryHarvest["quiescent"],secondaryHarvest["quiescent"],tertiaryHarvest["quiescent"]])
    return{"stem":S, "progenitor":P, "differentiated":D, "quiescent":Q}

def multiGrowthSimulation(testParameters, quiescent):
    simulation = 10000
    if quiescent == True:
        s = np.random.binomial(n=simulation, p=0.17)
        p = np.random.binomial(n=simulation, p=0.64)
        timeRange = int(len(countingTimePoints))
        multiGrowth = np.zeros((simulation,4,timeRange))
        for i in range(simulation):
            if i < s:
                clone = passageExpansion({"stem":1,"progenitor":0,"quiescent":0}, testParameters, 0.37)
            elif i < (s + p):
                clone = passageExpansion({"stem":0,"progenitor":1,"quiescent":0}, testParameters, 0.37)
            else:
                clone = passageExpansion({"stem":0,"progenitor":0,"quiescent":1}, testParameters, 0.37)
            multiGrowth[i][0][:] = clone["stem"]
            multiGrowth[i][1][:] = clone["progenitor"]
            multiGrowth[i][2][:] = clone["differentiated"]
            multiGrowth[i][3][:] = clone["quiescent"]
    else:
        s = np.random.binomial(n=simulation, p=0.15)
        timeRange = int(len(countingTimePoints))
        multiGrowth = np.zeros((simulation,3,timeRange))
        for i in range(simulation):
            if i < s:
                clone = passageExpansion({"stem":1,"progenitor":0}, testParameters, 0.37)
            else:
                clone = passageExpansion({"stem":0,"progenitor":1}, testParameters, 0.37)
            multiGrowth[i][0][:] = clone["stem"]
            multiGrowth[i][1][:] = clone["progenitor"]
            multiGrowth[i][2][:] = clone["differentiated"]
    return(multiGrowth)

def barFreqDistribution(testParameters, quiescent):
    multiGrowth = multiGrowthSimulation(testParameters, quiescent)
    clones = len(multiGrowth)
    primary = np.zeros(clones)
    secondary = np.zeros(clones)
    tertiary = np.zeros(clones)
    for clone in range(clones):
        primary[clone] = sum(np.array([multiGrowth[clone][cellType][0] for cellType in range(3)]))
        secondary[clone] = sum(np.array([multiGrowth[clone][cellType][1] for cellType in range(3)]))
        tertiary[clone] = sum(np.array([multiGrowth[clone][cellType][2] for cellType in range(3)]))
    primaryBarFreq = primary/sum(primary)
    secondaryBarFreq = secondary/sum(secondary)
    tertiaryBarFreq = tertiary/sum(tertiary)
    return{"primary":primaryBarFreq, "secondary":secondaryBarFreq, "tertiary": tertiaryBarFreq}
    