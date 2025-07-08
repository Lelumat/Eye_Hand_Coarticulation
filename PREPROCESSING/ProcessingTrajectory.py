import pandas as pd
import Utilities
import numpy as np
import json
import copy as cp
import scipy.interpolate as interpolate
import Utilities



with open("./OUTPUT/subjNames.txt", "r") as f:
    subjNames = f.read().splitlines()
with open("./OUTPUT/problemList.txt", "r") as f:
    problemList = f.read().splitlines()

for subjName in subjNames:
    print("SUBJ: ", subjName)
    df = pd.read_csv("./ OUTPUT/" + subjName +"dfSampleVis.csv", low_memory=False)
    print("Imported Subject data\n")
    xLs, yLs, motifLs, relMotifLs, fixX, fixY, LABEL_FIXLs, TRIAL_INDEXLs, nextMotifLs = [], [], [], [], [], [], [], [], []
    
    for TRIAL_INDEX in range(94):
        print("TRIAL_INDEX: ", TRIAL_INDEX)
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        movementMask = df.BEFORE_MOVEMENT == False
        draggingMask = df.DRAGGING == True
        problem = df.loc[trialMask & movementMask, "MAPNAME"].values[0]
        touchMask = df.TYPE == "TOUCH"
        noSaccaMask = df.LABEL_FIX.notna()

        #Loop over LABEL_FIX
        for LABEL_FIX in df.loc[trialMask & movementMask &  noSaccaMask, "LABEL_FIX"].unique():
            #print("LABEL_FIX: ", LABEL_FIX)
            #print("LABEL_FIX: ", LABEL_FIX)
            labelFixMask = df.LABEL_FIX == LABEL_FIX
            #If theare are rows in labelFixMask with PAUSE == True, then skip the LABEL_FIX
            if df.loc[trialMask & movementMask & labelFixMask & touchMask, "PAUSE"].any():
                #print("     PAUSE")
                continue
            #If there are rows with DRAGGING == False, then skip the LABEL_FIX
            if df.loc[trialMask & movementMask & labelFixMask & touchMask, "DRAGGING"].any() == False:
                #print("     DRAGGING")
                continue
            #If there are rows with MISSING_TOUCH == True, then skip the LABEL_FIX 
            if df.loc[trialMask & movementMask & labelFixMask & touchMask, "MISSING_TOUCH"].any():
                #print("     MISSING TOUCH")
                continue
            #f there are rows with BACKTRACK == True, then skip the LABEL_FIX
            if df.loc[trialMask & movementMask & labelFixMask & touchMask, "BACKTRACK"].any():
                #print("     BACKTRACK")
                continue
            #Get the motif
            relMotif = df.loc[trialMask & labelFixMask & touchMask, "relMotif"].values[0]
            motif = df.loc[trialMask & labelFixMask & touchMask, "motif"].values[0]
            if motif == "NOTHING":
                continue
            try:
                nextMotif = df.loc[trialMask & (df.LABEL_FIX == LABEL_FIX + 1) & touchMask, "motif"].values[0]
            except:
                nextMotif = "NOTHING"
           
            #print("ACCEPTED\n")
            #Get the reference node:
            startNode = Utilities.parse_string(df.loc[trialMask & draggingMask & movementMask & labelFixMask & touchMask, "almostActiveNode"].values[0])[0]
            
            #Get the angle
            angle = df.loc[trialMask & movementMask & labelFixMask & touchMask, "angle"].values[0]
            
            #Get the trajectory
            x = df.loc[trialMask & movementMask & draggingMask & labelFixMask & touchMask, "x"].values
            y = df.loc[trialMask & movementMask & draggingMask & labelFixMask & touchMask, "y"].values
            t = df.loc[trialMask & movementMask & draggingMask & labelFixMask & touchMask, "t"].values

            xOrigin, yOrigin = Utilities.mapGraph2Screen(problem, [(startNode[0], startNode[1])])
            xshifted = x - xOrigin
            yshifted = y - yOrigin
            trajectory = list(zip(xshifted, yshifted))

            #Rotate the trajectory   
            trajectory = Utilities.rotateFloat(angle, trajectory)
            xroto_shifted = cp.deepcopy([x for (x, y) in trajectory])
            yroto_shifted = cp.deepcopy([y for (x, y) in trajectory]) 

            #Add a small noise to the trajectory
            xroto_shifted = xroto_shifted + np.random.normal(0, 0.0000001, len(xroto_shifted))
            yroto_shifted = yroto_shifted + np.random.normal(0, 0.0000001, len(yroto_shifted))

            #SPLINE FITTING
            tck, u = interpolate.splprep([xroto_shifted, yroto_shifted], k = 1, s=0)

            #RESAMPLING
            #VELOCITY RESAMPLING
            Dist = np.sqrt(np.diff(xroto_shifted)**2 + np.diff(yroto_shifted)**2)
            indexes = np.linspace(0, len(Dist)-1, 100)
            Dist = Dist[indexes.astype(int)]
            #Normalize dVt such that the sum of all the elements is 1
            unew = np.cumsum(Dist/np.sum(Dist))
            #UNIFORM RESAMPLING
            #unew = np.linspace(0, 1, 100)
            resampledTrajectory = interpolate.splev(unew, tck)
            xresampled = resampledTrajectory[0]
            yresampled = resampledTrajectory[1]

            #Get the average eye position
            avgx = df.loc[trialMask & movementMask & draggingMask & labelFixMask, "avgx"].values[0]
            avgy = df.loc[trialMask & movementMask & draggingMask & labelFixMask, "avgy"].values[0]

            #Shift  avgx abd avgy
            avgx = avgx - xOrigin
            avgy = avgy - yOrigin

            #Rotate the average eye position
            avgx, avgy = Utilities.rotateFloat(angle, [(avgx, avgy)])[0]

            #Find the angle of the average eye position w.r.t. the last node of the motif  TBC
            endNode = Utilities.parse_string(df.loc[trialMask & draggingMask & movementMask & labelFixMask & touchMask, "almostActiveNode"].values[0])[-1]
            xEnd, yEnd = Utilities.mapGraph2Screen(problem, [(endNode[0], endNode[1])])
            avgx2 = avgx - xEnd
            avgy2 = avgy - yEnd
            angle2 = np.arctan2(avgy2, avgx2)
            #Get the posizion of the next fixation
            nextFixX = df.loc[trialMask & movementMask & draggingMask & (df.LABEL_FIX == LABEL_FIX + 1), "avgx"].values[0]
            nextFixY = df.loc[trialMask & movementMask & draggingMask & (df.LABEL_FIX == LABEL_FIX + 1), "avgy"].values[0]
            nextFixX = nextFixX - xEnd
            nextFixY = nextFixY - yEnd
            nextFixAngle = np.arctan2(nextFixY, nextFixX)

            #Update lists
            xLs += list(xresampled)
            yLs += list(yresampled)
            relMotifLs += [relMotif]*len(xresampled)
            motifLs += [motif]*len(xresampled)
            fixX += [avgx]*len(xresampled)
            fixY += [avgy]*len(xresampled)
            LABEL_FIXLs += [LABEL_FIX]*len(xresampled)
            TRIAL_INDEXLs += [TRIAL_INDEX]*len(xresampled)
            nextMotifLs += [nextMotif]*len(xresampled)
  
    #Save the data
    df = pd.DataFrame({"x": xLs, "y": yLs, "relMotif": relMotifLs, "motif": motifLs, "fixX": fixX, "fixY": fixY, "LABEL_FIX": LABEL_FIXLs, "TRIAL_INDEX": TRIAL_INDEXLs, "nextMotif": nextMotifLs})
    df["relNextMotif"] = df.apply(lambda x: Utilities.relativize(x["motif"] + x["nextMotif"]) if x["motif"] != "NOTHING"  and x["nextMotif"] != "NOTHING" else "NOTHING", axis=1)
    df["CLASS"] = df[["nextMotif","relNextMotif"]].apply(lambda x: x[1 + len(nextMotif["nextMotif"])+1] if x != "NOTHING" else "NOTHING")  
    df.to_csv("./OUTPUT/" + subjName + "dfTrajectoryVisResampled.csv", index=False)