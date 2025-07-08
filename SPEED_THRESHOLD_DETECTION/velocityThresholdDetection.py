####DOVRESTI FARE ALMENO UNA VALIDAZIONE A OCCHIO SUGLI ISTOGRAMMI

import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#Import the subjNames
with open("./OUTPUT/subjNames.txt", "r") as f:
    subjNames = f.read().splitlines()

thresholdDict = {}

for subjName in subjNames:
    print("subjName: ", subjName)
    df = pd.read_csv("./OUTPUT/" + subjName +"dfSampleVis.csv")
    hist, bin_edges = np.histogram(df.vMouse[~np.isnan(df.vMouse)], bins=50, range = (0, 1.5))
    minima = np.r_[True, hist[1:] <= hist[:-1]] & np.r_[hist[:-1] <= hist[1:], True]
    minima = np.where(minima == True)
    minima = minima[0][0]
    threshold = bin_edges[minima]
    thresholdDict[subjName] = threshold
#Save the dictionary as a json file

with open('./OUTPUT/thresholdDict.json', 'w') as fp:
    json.dump(thresholdDict, fp)
