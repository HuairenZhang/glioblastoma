import numpy as np

def setConstantBins(exp):
    binWidth = max(exp)/100
    bins = np.array([(binWidth*n) for n in range(101)])
    return(bins)

def binnedBarcodeFrequency(barFreq,bins):
    binnedBarFreq = np.zeros(len(bins)-1)
    for B in range(len(bins)-1):
        index = np.all([[bins[B] <= barFreq], [bins[B+1] > barFreq]], axis = 0)
        binnedBarFreq[B] = sum(index[0])/len(barFreq)
    return(binnedBarFreq)

def normalisation(probability):
    return(probability/sum(probability))

