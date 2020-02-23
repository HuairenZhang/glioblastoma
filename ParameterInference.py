# Import modules
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import math
import pickle
import pyabc
from pyabc import (ABCSMC,RV, Distribution, MedianEpsilon, transition, epsilon, sampler, AdaptivePNormDistance)
from pyabc.visualization import (plot_kde_1d, plot_kde_2d, plot_kde_matrix, plot_epsilons, plot_effective_sample_sizes)

# Parameter inference for stem cell hierarchy model
from StemCellHierarchyModel import clonalGrowth
# Parameter inference for quiescent cell dynamics model
# parameters = {"k1":0.0225,"k2":0.1275,"lambda":1.0,"k7":0.48}
# from QuiescentCellDynamicsModel import clonalGrowth

from SerialTransplantation import (transplantation, passageExpansion, multiGrowthSimulation, barFreqDistribution)
from Binning import (setConstantBins, binnedBarcodeFrequency, normalisation)
from Distance import (Hellinger, measureDistance, DistanceAfterBinning)

# Import experimental data
countingTimePoints = np.array([80,65,70])
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

# The average distribution of barcode frequency is used for parameter inference
primaryExp = np.array([np.mean(i) for i in np.transpose(primaryExpBarFreq)])
secondaryExp = np.array([np.mean(i) for i in np.transpose(secondaryExpBarFreq)])
tertiaryExp = np.array([np.mean(i) for i in np.transpose(tertiaryExpBarFreq)])
priBins = setConstantBins(primaryExp)
secBins = setConstantBins(secondaryExp)
terBins = setConstantBins(tertiaryExp)
primaryExpBinned = binnedBarcodeFrequency(primaryExp,priBins)
secondaryExpBinned = binnedBarcodeFrequency(secondaryExp,secBins)
tertiaryExpBinned = binnedBarcodeFrequency(tertiaryExp,terBins)
primaryExpNorm = normalisation(primaryExpBinned[1:])
secondaryExpNorm = normalisation(secondaryExpBinned[1:])
tertiaryExpNorm = normalisation(tertiaryExpBinned[1:])
exp = np.array([normalisation(primaryExpBinned[1:]),normalisation(secondaryExpBinned[1:]), normalisation(tertiaryExpBinned[1:])])
expBarFreq = {"barcodeFrequency":exp}

# The simulated barcode frequencies are also binned
def determineTestParameters(testParameters):
    quiescent = False
    barFreq = barFreqDistribution(testParameters, quiescent)
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

# Generate synthetic data
# simBarFreq = determineTestParameters({"Omega":0.15,"Probability":0.15,"Lambda":1.0,"Gamma":0.48})

# Parameter inference using approximate Bayesian computation (pyABC)
limits = dict(Omega = (0, 0.3), Probability = (0, 0.2), Lambda = (0, 1.5), Gamma = (0, 3))
parameter_prior = Distribution(**{key: RV("uniform", a, b - a) for key, (a,b) in limits.items()})
db_path = pyabc.create_sqlite_db_id(file_ = "glioblatomaLanModel_syn.db")
abc = ABCSMC(models = determineTestParameters, \
    parameter_priors = parameter_prior, \
    distance_function = DistanceAfterBinning, \
    population_size = 160, \
    sampler = sampler.MulticoreParticleParallelSampler(), \
    transitions = transition.LocalTransition(k_fraction=0.3))
abc.new(db_path, expBarFreq);
h = abc.run(minimum_epsilon=0.1, max_nr_populations=10)

df, w = h.get_distribution(m=0)
plot_kde_matrix(df, w, limits=limits)
plt.savefig('infer_result.pdf')
plt.clf()

pickle_out = open('result','wb')
pickle.dump(h.get_distribution(), pickle_out)
pickle_out.close()

plot_epsilons(h)
plt.savefig('epsilon_plot.pdf')
plt.clf()
