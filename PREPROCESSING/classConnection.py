import json
import Utilities
import pandas as pd

with open("./OUTPUT/subjNames.txt", "r") as f:
    subjNames = f.read().splitlines()
with open("./OUTPUT/problemList.txt", "r") as f:
    problemList = f.read().splitlines()
with open("./OUTPUT/graphNodeIdDict.json", "r") as f:
    graphNodeIdDict = json.load(f)

for subjName in subjNames:
    print(subjName)
    df = pd.read_csv("./OUTPUT/" + subjName +"dfSampleVis.csv", low_memory=False)
    dfSubj = pd.read_csv("./OUTPUT/" + subjName +"dfTrajectoryVisResampled.csv", low_memory=False)
    
    touchMask = df["TYPE"] == "TOUCH"

    #Iterate over TRIAL_INDEX and LABEL_FIX
    for TRIAL_INDEX in dfSubj["TRIAL_INDEX"].unique():
        trial_mask = df["TRIAL_INDEX"] == TRIAL_INDEX
        for LABEL_FIX in df.loc[df.TRIAL_INDEX == TRIAL_INDEX, "LABEL_FIX"].unique()[:-1]:
            label_mask = df["LABEL_FIX"] == LABEL_FIX
            next_label_mask = df["LABEL_FIX"] == LABEL_FIX + 1
            
            motif = dfSubj.loc[(dfSubj["TRIAL_INDEX"] == TRIAL_INDEX) & (dfSubj["LABEL_FIX"] == LABEL_FIX), "motif"].values[0]
            
            motive_nodes1 = Utilities.parse_string(df[trial_mask & label_mask & touchMask]["motiveNodes"].values[-1])
            motive_nodes2 = Utilities.parse_string(df[trial_mask & next_label_mask & touchMask]["motiveNodes"].values[0])
        
            last_node1 = motive_nodes1[-1]
            first_node2 = motive_nodes2[0]
        
            if last_node1 != first_node2:
                try:
                    next_motif = Utilities.convertIntoMotif([last_node1, first_node2])[0]
                except:
                    condition = (dfSubj["LABEL_FIX"] == LABEL_FIX) & (dfSubj["TRIAL_INDEX"] == TRIAL_INDEX)
                    dfSubj.loc[condition, "CLASS"] = "NOTHING"
                    continue
                new_class = Utilities.relativize(motif + next_motif)[len(motif):len(motif) + 1]
                condition = (dfSubj["LABEL_FIX"] == LABEL_FIX) & (dfSubj["TRIAL_INDEX"] == TRIAL_INDEX)
                dfSubj.loc[condition, "CLASS"] = new_class
    #save dfSubj
    dfSubj.to_csv("./OUTPUT/" + subjName + "dfTrajectoryVisResampled.csv", index=False)