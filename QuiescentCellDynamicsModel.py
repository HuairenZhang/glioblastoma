import random
import numpy as np

def clonalGrowth(inject, testParameters, time):
    k1 = parameters["k1"]
    k2 = parameters["k2"]
    k3 = testParameters["k3"]
    k4 = parameters["lambda"]*0.5
    k5 = parameters["lambda"]*0.5
    k6 = testParameters["k6"]
    k7 = parameters["k7"]
    k8 = testParameters["k8"]
    S = inject["stem"]
    P = inject["progenitor"]
    D = 0
    Q = inject["quiescent"]
    reactionTime = 0      
    cont = True  
    while(cont):
        A1 = S*k1
        A2 = S*k2
        A3 = S*k3
        A4 = P*k4
        A5 = P*k5
        A6 = P*k6
        A7 = D*k7
        A8 = Q*k8
        A0 = A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8
        if A0 > 0:
    #         first random number generator to determine the time of nexr reaction
            r1 = random.random()
            t = np.log(1/r1)/A0
            if reactionTime + t > time:
                cont = False
            else:
                reactionTime += t
    #             second random number generator to determine which reaction occurs
                r2 = random.random()
                if r2 < A1/A0:
                    S += 1
                elif r2 < (A1+A2)/A0:
                    P += 1
                elif r2 < (A1+A2+A3)/A0:
                    S -= 1
                elif r2 < (A1+A2+A3+A4)/A0:
                    P += 1
                elif r2 < (A1+A2+A3+A4+A5)/A0:
                    P -= 1
                    D += 1
                elif r2 < (A1+A2+A3+A4+A5+A6)/A0:
                    P -= 1
                    Q += 1
                elif r2 < (A1+A2+A3+A4+A5+A6+A7)/A0:
                    D -= 1
                else:
                    S += 1
                    Q -=1
        else:
            cont = False
    return{"stem":S,"progenitor":P,"differentiated":D,"quiescent":Q}
