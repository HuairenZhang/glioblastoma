import math

def Hellinger(o,m):
    return(math.sqrt(sum([(math.sqrt(a)-math.sqrt(b))**2 for a,b in zip(o,m)])/2))

def measureDistance(f,sim,data):
    results = [f(sim[i],data[i]) for i in range(len(sim))]
    return(sum(results))

def DistanceAfterBinning(sim,exp):
    simBarFreq = sim["barcodeFrequency"]
    expBarFreq = exp["barcodeFrequency"]
    result = measureDistance(Hellinger, simBarFreq, expBarFreq)
    return(result)