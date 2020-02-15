import random
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from scipy import stats
import pandas as pd
import math
from functools import partial
import pickle
import pyabc
from pyabc import (ABCSMC,RV, Distribution, MedianEpsilon, transition, epsilon, sampler, AdaptivePNormDistance)
from pyabc.visualization import (plot_kde_1d, plot_kde_2d, plot_kde_matrix, plot_epsilons, plot_effective_sample_sizes)

countingTimePoints = np.array([80,65,70])

def clonalGrowth(inject, parameters, time):
    k1 = parameters["Omega"] * parameters["Probability"]
    k2 = parameters["Omega"] * (1 - parameters["Probability"])
    k3 = parameters["Lambda"]*0.5
    k4 = parameters["Lambda"]*0.5
    k5 = parameters["Gamma"]
    S = inject["stem"]
    P = inject["progenitor"]
    D = 0
    reactionTime = 0
    cont = True
    while(cont):
        A1 = S*k1
        A2 = S*k2
        A3 = P*k3
        A4 = P*k4
        A5 = D*k5
        A0 = A1 + A2 + A3 + A4 + A5
#         first random number generator to determine the time of nexr reaction\
        if A0 > 0:
            r1 = random.random()
            t = np.log(1/r1)/A0
            if reactionTime + t > time:
                cont = False
            else:
                reactionTime += t
    #             second random number generator to determine which reaction occurs\
                r2 = random.random()
                if r2 < A1/A0:
                    S += 1
                elif r2 < (A1+A2+A3)/A0:
                    P += 1
                elif r2 < (A1+A2+A3+A4)/A0:
                    P -= 1
                    D += 1
                else:
                    D -= 1
        else:
            cont = False
    return{"stem":S,"progenitor":P,"differentiated":D}

def transplantation(harvest,transplantationProb):
    stemHarvest = harvest["stem"]
    proHarvest = harvest["progenitor"]
    totalHarvest = stemHarvest + proHarvest
    stemReinject = 0
    proReinject = 0
    quieReinject = 0
    if totalHarvest > 0:
        stemProb = stemHarvest/totalHarvest
        proProb = proHarvest/totalHarvest
        stemReinject += np.random.binomial(n=totalHarvest, p=stemProb*transplantationProb)
        proReinject += np.random.binomial(n=totalHarvest, p=proProb*transplantationProb)
    else:
        pass
    return{"stem":stemReinject, "progenitor":proReinject}

def passageExpansion(primaryInject, testParameters, transplantationProb):
    primaryHarvest = clonalGrowth(primaryInject, testParameters,countingTimePoints[0])
    secondaryInject = transplantation(primaryHarvest,transplantationProb)
    secondaryHarvest = clonalGrowth(secondaryInject, testParameters,countingTimePoints[1])
    tertiaryInject = transplantation(secondaryHarvest,transplantationProb)
    tertiaryHarvest = clonalGrowth(tertiaryInject, testParameters,countingTimePoints[2])
    S = np.array([primaryHarvest["stem"],secondaryHarvest["stem"],tertiaryHarvest["stem"]])
    P = np.array([primaryHarvest["progenitor"],secondaryHarvest["progenitor"],tertiaryHarvest["progenitor"]])
    D = np.array([primaryHarvest["differentiated"],secondaryHarvest["differentiated"],tertiaryHarvest["differentiated"]])
    return{"stem":S, "progenitor":P, "differentiated":D}

def multiGrowthSimulation(testParameters):
    simulation = 10000
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

def barFreqDistribution(testParameters):
    multiGrowth = multiGrowthSimulation(testParameters)
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

experiment = pd.read_excel("Table_1_experimental_clonal_size.xlsx")
primaryExpBarFreq = np.vstack([experiment["(1)719 Ipsi"], experiment["(1)719 Contra"]])
secondaryExpBarFreq = np.vstack([\
#     experiment["(1,1V)719 Ipsi"], experiment["(1,1V)719 Contra"],\
                       experiment["(1,2V)719 Ipsi"], experiment["(1,2V)719 Contra"],\
#                        experiment["(1,3V)719 Ipsi"], experiment["(1,3V)719 Contra"],\
#                        experiment["(1,1T)719 Ipsi"], experiment["(1,1T)719 Contra"],\
#                        experiment["(1,2T)719 Ipsi"], experiment["(1,2T)719 Contra"],\
#                        experiment["(1C,1)719 Ipsi"], experiment["(1C,1)719 Contra"],\
#                        experiment["(1C,2)719 Ipsi"], experiment["(1C,2)719 Contra"]\
                      ])
tertiaryExpBarFreq = np.vstack([experiment["(1,2V,1)719 Ipsi"], experiment["(1,2V,1)719 Contra"]\
#                       ,experiment["(1,2V,2)719 Ipsi"], experiment["(1,2V,2)719 Contra"],\
#                       experiment["(1,2V,3)719 Ipsi"], experiment["(1,2V,3)719 Contra"],\
#                       experiment["(1,3V,1V)719 Ipsi"], experiment["(1,3V,1V)719 Contra"],\
#                       experiment["(1,3V,2V)719 Ipsi"], experiment["(1,3V,2V)719 Contra"],\
#                       experiment["(1,3V,1T)719 Ipsi"], experiment["(1,3V,1T)719 Contra"],\
#                       experiment["(1,1T,1V)719 Ipsi"], experiment["(1,1T,1V)719 Contra"],\
#                       experiment["(1,1T,2V)719 Ipsi"], experiment["(1,1T,2V)719 Contra"],\
#                       experiment["(1,1T,1T)719 Ipsi"], experiment["(1,1T,1T)719 Contra"],\
#                       experiment["(1,1T,2T)719 Ipsi"], experiment["(1,1T,2T)719 Contra"],\
#                       experiment["(1,1T,3T)719 Ipsi"], experiment["(1,1T,3T)719 Contra"]\
                     ])

primaryExp = np.array([np.mean(i) for i in np.transpose(primaryExpBarFreq)])
secondaryExp = np.array([np.mean(i) for i in np.transpose(secondaryExpBarFreq)])
tertiaryExp = np.array([np.mean(i) for i in np.transpose(tertiaryExpBarFreq)])

def setConstantBins(exp):
    binWidth = max(exp)/100
    bins = np.array([(binWidth*n) for n in range(101)])
    return(bins)

priBins = setConstantBins(primaryExp)
secBins = setConstantBins(secondaryExp)
terBins = setConstantBins(tertiaryExp)

def binnedBarcodeFrequency(barFreq,bins):
    binnedBarFreq = np.zeros(len(bins)-1)
    for B in range(len(bins)-1):
        index = np.all([[bins[B] <= barFreq], [bins[B+1] > barFreq]], axis = 0)
        binnedBarFreq[B] = sum(index[0])/len(barFreq)
    return(binnedBarFreq)

primaryExpBinned = binnedBarcodeFrequency(primaryExp,priBins)
secondaryExpBinned = binnedBarcodeFrequency(secondaryExp,secBins)
tertiaryExpBinned = binnedBarcodeFrequency(tertiaryExp,terBins)

def normalisation(probability):
    return(probability/sum(probability))

primaryExpNorm = normalisation(primaryExpBinned[1:])
secondaryExpNorm = normalisation(secondaryExpBinned[1:])
tertiaryExpNorm = normalisation(tertiaryExpBinned[1:])

exp = np.array([normalisation(primaryExpBinned[1:]),normalisation(secondaryExpBinned[1:]), normalisation(tertiaryExpBinned[1:])])
expBarFreq = {"barcodeFrequency":exp}

def determineTestParameters(testParameters):
    barFreq = barFreqDistribution(testParameters)
    primaryBarFreq = barFreq["primary"]
    secondaryBarFreq = barFreq["secondary"]
    tertiaryBarFreq = barFreq["tertiary"]
    primaryBinned = binnedBarcodeFrequency(primaryBarFreq,priBins)
    secondaryBinned = binnedBarcodeFrequency(secondaryBarFreq,secBins)
    tertiaryBinned = binnedBarcodeFrequency(tertiaryBarFreq,terBins)
    primaryNorm = normalisation(primaryBinned[1:])
    secondaryNorm = normalisation(secondaryBinned[1:])
    tertiaryNorm = normalisation(tertiaryBinned[1:])
    return{"barcodeFrequency": np.array([primaryNorm, secondaryNorm, tertiaryNorm])}

def Hellinger(o,m):
    return(math.sqrt(sum([(math.sqrt(a)-math.sqrt(b))**2 for a,b in zip(o,m)])/2))

def measureDistance(f,sim,data):
    results = [f(sim[i],data[i]) for i in range(len(sim))]
    return(sum(results))

def DistanceAfterBinning(sim,exp):
    f = Hellinger
    simBarFreq = sim["barcodeFrequency"]
    expBarFreq = exp["barcodeFrequency"]
    result = measureDistance(Hellinger, simBarFreq, expBarFreq)
    return(result)

limits = dict(Omega = (0, 0.3), Probability = (0, 0.2), Lambda = (0, 1.5), Gamma = (0, 3))
parameter_prior = Distribution(**{key: RV("uniform", a, b - a) for key, (a,b) in limits.items()})
db_path = pyabc.create_sqlite_db_id(file_ = "glioblatomaLanModel_syn.db")
abc = ABCSMC(models = determineTestParameters, \
    parameter_priors = parameter_prior, \
    distance_function = DistanceAfterBinning, \
    population_size = 240, \
    sampler = sampler.MulticoreParticleParallelSampler(), \
    transitions = transition.LocalTransition(k_fraction=0.3))

abc.new(db_path, expBarFreq);
h = abc.run(minimum_epsilon=0.5, max_nr_populations=3)

df, w = h.get_distribution(m=0)
plot_kde_matrix(df, w, limits=limits)
plt.savefig('infer_result.pdf')
plt.clf()

pickle_out = open('history','wb')
pickle.dump(h, pickle_out)
pickle_out.close()

pickle_out = open('result','wb')
pickle.dump(h.get_distribution(), pickle_out)
pickle_out.close()

plot_epsilons(h)
plt.savefig('epsilon_plot.pdf')
plt.clf()
