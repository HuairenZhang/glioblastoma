import random
import numpy as np

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
    return{"stem":S,"progenitor":P,"differentiated":D, "quiescent":0}
