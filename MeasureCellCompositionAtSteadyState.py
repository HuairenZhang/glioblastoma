parameters = {"stem_div_rate":0.15,"epsilon":0.15,"pro_div_rate":1.0,"apoptosis_rate":0.48}

def stationaryCellComposition(parameters):
    Omega = parameters["stem_div_rate"]
    Epsilon = parameters["epsilon"]
    Lambda = parameters["pro_div_rate"]
    Gamma = parameters["apoptosis_rate"]
    Ome = 1 + 0.5 * Lambda * (1 - Epsilon) / (Gamma + Epsilon * Omega)
    # stem cell
    S = Epsilon / Ome
    # progenitor cell
    P = (1 - Epsilon) / Ome
    # differentiated cell
    # D = 1 - (S + P)
    return{"stem":S, "progenitor":P}

stemPrimaryProb = stationaryCellComposition(parameters)["stem"]/(stationaryCellComposition(parameters)["stem"]+stationaryCellComposition(parameters)["progenitor"])

def transplantationProb(parameters,n,meanExpansion):
    S = stationaryCellComposition(parameters)["stem"]
    P = stationaryCellComposition(parameters)["progenitor"]
    prob = n/((S + P) * meanExpansion)
    return(prob)

transplantationProb = transplantationProb(parameters,2,10)
