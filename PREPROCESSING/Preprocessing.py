import pandas as pd
import numpy as np
import json
import getActiveNodeHistory as ganh
from scipy.signal import butter, filtfilt
import copy as cp
import Utilities
import sys
import time

print("IMPORTING DATA")
#IMPORT
with open("./OUTPUT/subjNames.txt", "r") as f:
    subjNames = f.read().splitlines()
with open("./OUTPUT/problemList.txt", "r") as f:
    problemList = f.read().splitlines()
with open("./OUTPUT/mapIndex2Problem.json", "r") as f:
    mapIndex2Problem = json.load(f)
with open("./OUTPUT/kinematicReparsing.json", "r") as f:
    kinematicReparsing = json.load(f)
with open("./OUTPUT/thresholdDict.json", "r") as f:
    thresholdsDict = json.load(f)
with open("./OUTPUT/startNodeNeighboors.json", "r") as f:
    startNodeNeighbors = json.load(f)
with open("./OUTPUT/originCoords.json", "r") as f:
    originCoords = json.load(f)
with open("./OUTPUT/graphNodeIdDict.json", "r") as f:
    graphNodeIdDict = json.load(f)

#SYS PARAMS: planning duration in seconds
#minimumPlanningThreshold = float(sys.argv[1])   #ms in the last simulations 1*60 indexes, rather than ms so 300 ms in total 
#planningIndexThreshold = int(minimumPlanningThreshold/17) #17 ms is the sampling frequency of the eye tracker

mapPath = "/Users/Lelumat/Desktop/EYE_TRACKER/MAPS/"
level1Origin = [ 284, 1920 - 1215]
level2Origin = [ 217, 1920 - 1282]
level3Origin = [ 217, 1920 - 1415]
linkEstimatedLenght = 132.65217391304347
actions2cardinal = {(0,1): "N", (0,-1): "S", (1,0): "E", (-1,0): "W"}
angleDict = {"N": 0, "E":90, "S":180, "W":270}
#import all the graphs at once
graphLs = [Utilities.map_load(mapPath + problem[:-5]) for problem in problemList]
mapDict = {problem:graphLs[problemList.index(problem)] for problem in problemList}

t0 = time.time()
for subjName in subjNames:
    t1 = time.time()
    fpath = "./DATA/SUBJ_" + subjName + "/"
    print("subjName: ", subjName)
    print("     STARTING EYE TRACKER DATA")




    
    ###################                                                 ###################
    #                                    EYE TRACKER DATA
    ###################                                                 ###################
    df =  pd.read_csv(fpath + subjName + "_SAMPLE_DATA.xls" ,sep="\t", decimal=",", low_memory=False, encoding = "utf-16")
    df["x"]= df["AVERAGE_GAZE_X"].str.replace(",",".").apply(lambda x: float(x) if x != "." else np.nan) 
    df["y"]= df["AVERAGE_GAZE_Y"].str.replace(",",".").apply(lambda x: float(x) if x != "." else np.nan)
    df["vx"] = df["AVERAGE_VELOCITY_X"].str.replace(",",".").apply(lambda x: float(x) if x != "." else np.nan)
    df["vy"] = df["AVERAGE_VELOCITY_Y"].str.replace(",",".").apply(lambda x: float(x) if x != "." else np.nan)
    df["v"] = np.sqrt(df["vx"]**2 + df["vy"]**2)
    df["SIZE"] = df["AVERAGE_PUPIL_SIZE"].str.replace(",",".").apply(lambda x: float(x) if x != "." else np.nan)
    df["t"] = df["SAMPLE_INDEX"]*0.5 -.5 #step di 0.5 ms per il campionamento di 2000 Hz
    #Let's make a column type with the value "FIX"
    df["TYPE"] = "FIX"
    #Let's mirror the y coordinates
    df["y"] = 1920 - df["y"]
    #Drop teh following columns ["AVERAGE_GAZE_X", "AVERAGE_GAZE_Y", "AVERAGE_PUPIL_SIZE", "SAMPLE_INDEX"]
    df.drop(columns=["AVERAGE_GAZE_X", "AVERAGE_GAZE_Y", "AVERAGE_PUPIL_SIZE", "SAMPLE_INDEX", "RECORDING_SESSION_LABEL", "AVERAGE_VELOCITY_X", "AVERAGE_VELOCITY_Y"], inplace=True)

    #resample only 1 point every 10
    resamplingFreq = 200 #Hz
    originalFreq = 2000  #Hz
    df = df.iloc[::int(originalFreq/resamplingFreq), :]
    df = df[df["TRIAL_INDEX"] != 1]
    #Subtract 2 from TRIAL_INDEX
    df["TRIAL_INDEX"] = df["TRIAL_INDEX"] - 2
    df = df.reset_index(drop=True)
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()






    ##############                                                        ##################                 
    #                                 MOUSE TRACKER DATA
    ##############                                                        ##################       
    print("     STARTING MOUSE TRACKER DATA")
    TRIAL_INDEX, tLs, xLs, yLs, activeNodeHistory = [], [], [], [], []
    files = mapIndex2Problem[subjName]
    for problem in problemList:
        with open(fpath + problem, "r") as f:
            data = json.load(f)
        #print("          problem:", problemList.index(problem))
        g = graphLs[problemList.index(problem)]
        #Let's extract from data["UserTrajectory"] the time and x and y coordinates
        tTrialData = [d["x"] for d in data["UserTrajectory"]] 
        tLs += tTrialData
        x = np.array([d["y"] for d in data["UserTrajectory"]])
        y = np.array([d["z"] for d in data["UserTrajectory"]])

        #tLs += [t for (t, x, y) in kinematicReparsing[subjName][problem]]
        trajectory = Utilities.mapExp2Screen(problem, x, y)
        xLs += [x_1 for (x_1, _) in trajectory]
        yLs += [y_1 for (_, y_1) in trajectory]
        TRIAL_INDEX += [int(list(files.values()).index(problem[:-5]))]*len(x)
        activeNodeHistory += [np.nan]*len(x)
        
        #Let's extract from data["UserTrajectory"] the time and x and y coordinates and move to screen coordinates
        mapId2Node = {i:n for i, n in enumerate(g.nodes())}
        fullNodePath = data["FullNodePath"]
        tEvents =  ganh.getValidTimeEvents(data["TrialEventData"])

        #Find the indexes of elems of tTrialData that are in tEvents
        indexes = [i for i in range(len(tLs) - len(tTrialData), len(tLs)) if tLs[i] in tEvents]
        
        #Remove the indexes from tLs, xLs, yLs, activeNodeHistory, TRIAL_INDEX
        tLs = [tLs[i] for i in range(len(tLs)) if i not in indexes]
        xLs = [xLs[i] for i in range(len(xLs)) if i not in indexes]
        yLs = [yLs[i] for i in range(len(yLs)) if i not in indexes]
        activeNodeHistory = [activeNodeHistory[i] for i in range(len(activeNodeHistory)) if i not in indexes]
        TRIAL_INDEX = [TRIAL_INDEX[i] for i in range(len(TRIAL_INDEX)) if i not in indexes]
        
        #Add the indexes of elems of tLs that are in tEvents
        tLs += tEvents
        xLs += [np.nan]*len(fullNodePath)
        yLs += [np.nan]*len(fullNodePath)  
        TRIAL_INDEX += [int(list(files.values()).index(problem[:-5]))] * len(fullNodePath)
        activeNodeHistory += [[mapId2Node[n] for n in nodesLs ] for nodesLs in ganh.get_active_nodes(fullNodePath)]

    #let's add to a dataframe these lists
    dfTouch = pd.DataFrame({"TRIAL_INDEX":TRIAL_INDEX , "t":tLs, "x":xLs, "y":yLs, "activeNode":activeNodeHistory})
    #Define the TYPE of the event
    dfTouch["TYPE"] = "TOUCH"
    #Convert the time in milliseconds
    dfTouch.t*=1000 
    #Sort the dataframe by TRIAL_INDEX and t
    dfTouch = dfTouch.sort_values(by=['TRIAL_INDEX', 't']).reset_index(drop=True)
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()

    ###############   ACTIVE NODE 1   ###############
    #Create a mask to identify the NaN values within each TRIAL_INDEX group
    mask = dfTouch.groupby("TRIAL_INDEX")["activeNode"].transform(lambda x: x.isna().cumsum() > 0)
    # Forward fill NaN values within the same TRIAL_INDEX group using the mask
    dfTouch.loc[mask, "activeNode"] = dfTouch.groupby("TRIAL_INDEX")["activeNode"].ffill()
    # Replace all the NaN values in dfMerge.activeNode by "None"
    dfTouch.loc[mask, "activeNode"] = dfTouch.activeNode.fillna("NOTHING")
    #let's make dfActiveNodes.activeNode a string
    dfTouch.loc[mask, "activeNode"] = dfTouch.activeNode.astype(str)
    
    ##################          KINEMATICS         #################
    print("     MOUSE TRACKER KINEMATICS")
    b, a = butter(2, .1, analog=False)

    for TRIAL_INDEX in range(94):
        mask = (dfTouch.TRIAL_INDEX == TRIAL_INDEX)
        #Interpolate the missing values of x and y with a linear method
        dfTouch.loc[mask, "x"] = dfTouch.loc[mask, "x"].interpolate(method="linear")
        dfTouch.loc[mask, "y"] = dfTouch.loc[mask, "y"].interpolate(method="linear")
        #Compute vx and vy
        dfTouch.loc[mask, "vMousex"] = np.gradient(dfTouch.loc[mask, "x"], dfTouch.loc[mask, "t"])
        dfTouch.loc[mask, "vMousey"] = np.gradient(dfTouch.loc[mask, "y"], dfTouch.loc[mask, "t"])
        
        #Mask with non NaN values
        mask = (dfTouch.TRIAL_INDEX == TRIAL_INDEX) & (~dfTouch.vMousex.isna())
        #filter vx and vy
        dfTouch.loc[mask, "vMousex"] = filtfilt(b, a, dfTouch.loc[mask, "vMousex"].values)
        dfTouch.loc[mask, "vMousey"] = filtfilt(b, a, dfTouch.loc[mask, "vMousey"].values)
        dfTouch.loc[mask, "vMouse"] = np.sqrt(dfTouch.loc[mask, "vMousex"]**2 + dfTouch.loc[mask, "vMousey"]**2)
    
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()

    
    
    ################    ACTIVE NODE 2   #################
    print("     ACTIVE NODES")
    #Let's create an additional column before propagating the active node information
    dfTouch = dfTouch.drop(dfTouch[dfTouch.t == 0].index).reset_index(drop=True)
    df = pd.concat([df, dfTouch]).sort_values(by=['TRIAL_INDEX', 't']).reset_index(drop=True)

    for TRIAL_INDEX in range(94):
        mask = (df.TRIAL_INDEX == TRIAL_INDEX)
        df.loc[mask, "activeNode"] = df.groupby("TRIAL_INDEX")["activeNode"].ffill()
        df.loc[mask, "activeNode"] = df.activeNode.fillna("NOTHING")
        df.loc[mask, "activeNode"] = df.activeNode.astype(str)
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()


    ##################          DRAGGING       #################
    print("     DETECTING DRAGGING")    
    df["DRAGGING"] = False
    for TRIAL_INDEX in range(0, 94):
        trialMask = df["TRIAL_INDEX"] == TRIAL_INDEX
        problem = mapIndex2Problem[subjName][str(TRIAL_INDEX)]
        with open("./DATA/SUBJ_" + subjName + "/" + problem +".json", "r") as f:
                trialEvents = json.load(f) 
        trialEvents = trialEvents["TrialEventData"]
        draggingIntervals = []
        for event in trialEvents:
            if event["name"] == "startDragging":
                draggingIntervals.append(event["time"]*1000)
            if event["name"] == "StopDragging":
                draggingIntervals.append(event["time"]*1000)
        draggingIntervals.append(event["time"]*1000)

        #Get a list of interval extremes from draggingIntervals
        draggingIntervals = [interval for interval in zip(draggingIntervals[0::2], draggingIntervals[1::2])]
        #the rows for which t is between the extremes of the intervals are True
        for interval in draggingIntervals:
            df.loc[trialMask & df.t.between(interval[0], interval[1]), "DRAGGING"] = True
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()



    ##################            LABELLING FINGER KINEMATICS        ################
    print("     LABELLING FINGER KINEMATICS")
    #BEFORE_MOVEMENT
    #Create a column named BEFORE_MOVEMENT
    df["BEFORE_MOVEMENT"] = False
    for TRIAL_INDEX in range(0, 94):
        trialMask = df["TRIAL_INDEX"] == TRIAL_INDEX
        movingMask = df[trialMask].TYPE == "TOUCH"
        tStart = df[trialMask & movingMask].t.iloc[0]
        df.loc[trialMask & (df.t < tStart), "BEFORE_MOVEMENT"] = True
        df["BEFORE_MOVEMENT"].fillna(method='ffill', inplace=True)
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()


    ################# DETECT SLOW_MOTION #################
    print("     DETECTING SLOW MOTION")
    for TRIAL_INDEX in range(0, 94):
        trialMask = df["TRIAL_INDEX"] == TRIAL_INDEX
        movingMask = df[trialMask].TYPE == "TOUCH"
        df.loc[trialMask & movingMask, "SLOW_MOTION"] = df[trialMask & movingMask].vMouse < thresholdsDict[subjName]
        df["SLOW_MOTION"] = df["SLOW_MOTION"].fillna(method='ffill')
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()


    ################# DETECT PAUSE #################
    # Run the script on the synthetic DataFrame
    df["PAUSE_COUNT"] = np.nan
    df["PAUSE"] = np.nan
    pauseDurationLs = []
    velocityLs = []
    pauseDurationThreshold = 50
    for TRIAL_INDEX in range(94):
        trialMask = (df.TRIAL_INDEX == TRIAL_INDEX)
        touchMask = (df.TYPE == "TOUCH")
        dfTouch = df[trialMask & touchMask].reset_index(drop=False)
        dfTouch = dfTouch.rename(columns={'index': 'originalIndex'})
        indexesGroups = Utilities.split_into_consecutive_sublists(dfTouch[dfTouch.SLOW_MOTION == True].index)
        #Find the correspoding indexes of the df
        indexesGroups = [[dfTouch.loc[g[0], "originalIndex"], dfTouch.loc[g[-1], "originalIndex"]] for g in indexesGroups]
        #For all the rows in the indexesGroups, set the PAUSE to True and increse a counter
        for pause_count, g in enumerate(indexesGroups):
            #Compute the pause duration
            pause_duration = df.loc[g[1], "t"] - df.loc[g[0], "t"]
            #Compute the average velocity
            velocity = df.loc[g[0]:g[1], "vMouse"].mean()
            if pause_duration > pauseDurationThreshold:
                df.loc[g[0]:g[1], "PAUSE"] = True
                pauseDurationLs.append(pause_duration)
                velocityLs.append(velocity)
            else:
                df.loc[g[0]:g[1], "PAUSE"] = False
            df.loc[g[0]:g[1], "PAUSE_COUNT"] = pause_count 
    #Replace nan with False
    df["PAUSE"] = df["PAUSE"].fillna(False)    
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()

    




    ################# DETECT BEACKTRACK #################
    print("     DETECTING BACKTRACK")
    df["BACKTRACK"] = np.nan
    for TRIAL_INDEX in range(0, 94):
        trialMask = df["TRIAL_INDEX"] == TRIAL_INDEX
        #find consecutive rows whose activeNode is shorter than the previous one
        backtrackMask = df.loc[trialMask, "activeNode"].str.len() < df.loc[trialMask, "activeNode"].str.len().shift(1)
        #Set backtrackMask to True
        df.loc[trialMask & backtrackMask, "BACKTRACK"] = True
        forwardMask = df.loc[trialMask, "activeNode"].str.len() > df.loc[trialMask, "activeNode"].str.len().shift(1)
        df.loc[trialMask & forwardMask, "BACKTRACK"] = False
        df.loc[trialMask, "BACKTRACK"] = df.loc[trialMask, "BACKTRACK"].fillna(method='ffill')
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()
    




    ################# DETECT MISSING TOUCH #################
    print("     DETECTING MISSING TOUCH")
    #MISSING TOUCH
    df["MISSING_TOUCH"] = False
    for TRIAL_INDEX in range(0, 93):
        trialMask = df["TRIAL_INDEX"] == TRIAL_INDEX
        problem = mapIndex2Problem[subjName][str(TRIAL_INDEX)]
        with open("./DATA/SUBJ_" + subjName + "/" + problem +".json", "r") as f:
                trialEvents = json.load(f) 
        trialEvents = trialEvents["TrialEventData"]
        missed = 0
        missingTouch = []
        for i, event in enumerate(trialEvents):
            try:
                if event["edgeIsConnected"] == False and missed == 0:
                    #Missing touch starts
                    missed = 1
                    missingTouch.append(event["time"])
                    #print("start: ", missingTouch[-1])
                elif event["edgeIsConnected"] == True and missed == 1:
                    #Missing touch ends
                    missed = 0
                    missingTouch.append(trialEvents[i-1]["time"])
                    #print("end: ", missingTouch[-1])
            except:
                pass
        #Make a list of missing touch intervals
        missingTouchIntervals = [interval for interval in zip(missingTouch[0::2], missingTouch[1::2])]
        #Conservative interval for single node missing touch
        for inter in missingTouchIntervals:
            if inter[1] == inter[0]:
                #loop over trialEvents and fnid the index of the event with time == inter[0]
                missingIndex = [i for i, event in enumerate(trialEvents) if event["time"] == inter[0]][0]
                #Replace teh interval with one corresponding to the event before and the event after
                missingTouchIntervals[missingTouchIntervals.index(inter)] = (trialEvents[missingIndex - 1]["time"], trialEvents[missingIndex + 1]["time"])
        for interval in missingTouchIntervals:
            df.loc[trialMask & df.t.between(interval[0]*1000, interval[1]*1000), "MISSING_TOUCH"] = True
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()




    ################# DETECTING CLOSEST NODES #################
    print("     DETECTING CLOSEST NODES")
    #import the file again
    """dfCopy = pd.read_csv("./OUTPUT/" + subjName +"dfSampleVis.csv", low_memory=False)
    #Apply Utilities.parse_string to TOUCH_FIRST_CLOSEST_NODE and EYE_FIRST_CLOSEST_NODE 
    #Set TOUCH_FIRST_CLOSEST_NODE to the values of dfCopy
    df["TOUCH_FIRST_CLOSEST_NODE"] = dfCopy["TOUCH_FIRST_CLOSEST_NODE"]
    #Set EYE_FIRST_CLOSEST_NODE to the values of dfCopy
    df["EYE_FIRST_CLOSEST_NODE"] = dfCopy["EYE_FIRST_CLOSEST_NODE"]"""
    
    #MAPNAME
    df["MAPNAME"] = df["TRIAL_INDEX"].apply(lambda x: mapIndex2Problem[subjName][str(x)])
    #CLOSEST NODE TO THE EYES
    df.loc[df.TYPE == "FIX", "EYE_FIRST_CLOSEST_NODE"] = df.loc[df.TYPE == "FIX"].apply(lambda row: Utilities.find_closest_nodes([row.x, row.y], mapDict[row.MAPNAME + ".json"])[0][0], axis = 1)
    #CLOSEST NODE TO THE FINGER
    df.loc[df.TYPE == "TOUCH", "TOUCH_FIRST_CLOSEST_NODE"] = df.loc[df.TYPE == "TOUCH"].apply(lambda row: Utilities.find_closest_nodes([row.x, row.y], mapDict[row.MAPNAME + ".json"])[0][0], axis = 1)
    #Convert TOUCH_FIRST_CLOSEST_NODE and EYE_FIRST_CLOSEST_NODE to string
    df["TOUCH_FIRST_CLOSEST_NODE"] = df["TOUCH_FIRST_CLOSEST_NODE"].astype(str)
    df["EYE_FIRST_CLOSEST_NODE"] = df["EYE_FIRST_CLOSEST_NODE"].astype(str)
    
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()



    ################# DETECT FIXATIONS #################
    fixationVelocityThreshold = 9
    fixationDurationThreshold = 80 #ms
    fixationIndexThreshold = int(fixationDurationThreshold / 5)
    df["LABEL_FIX"] = np.nan
    df["FIXATION"] = np.nan
    df["FIXATION_DURATION"] = np.nan
    fixDurationLs = []
    velocityLs = []
    for TRIAL_INDEX in range(94):
        trialMask = (df.TRIAL_INDEX == TRIAL_INDEX)
        label_fix_count = 0
        fixMask = (df.TYPE == "FIX")
        dfFix = df[trialMask & fixMask].reset_index(drop=False)
        dfFix = dfFix.rename(columns={'index': 'originalIndex'})
        indexesGroups = Utilities.split_into_consecutive_sublists(dfFix[dfFix.v < fixationVelocityThreshold].index)
        #Find the correspoding indexes of the df
        indexesGroups = [[dfFix.loc[g[0], "originalIndex"], dfFix.loc[g[-1], "originalIndex"]] for g in indexesGroups]
        #For all the rows in the indexesGroups, set the FIXATION to True and increse a counter
        for fix_count, g in enumerate(indexesGroups):
            #Compute the FIXATION duration
            fix_duration = df.loc[g[1], "t"] - df.loc[g[0], "t"]
            #Compute the average velocity
            velocity = df.loc[g[0]:g[1], "v"].mean()
            if fix_duration > fixationDurationThreshold:
                df.loc[g[0]:g[1], "FIXATION"] = True
                #fixDurationLs.append(fix_duration)
                df.loc[g[0]:g[1], "FIXATION_DURATION"] = fix_duration  
                df.loc[g[0]:g[1], "LABEL_FIX"] = label_fix_count
                label_fix_count += 1  
                #velocityLs.append(velocity)
            else:
                df.loc[g[0]:g[1], "FIXATION"] = False
            #df.loc[g[0]:g[1], "LABEL_FIX"] = fix_count 
            #df.loc[g[0]:g[1], "FIXATION_DURATION"] = fix_duration     
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()


    ############   FIXATION AVERAGE COORDINATES    ############
    print("       ESTIMATE FIXATION COORDINATES")    
    df["avgx"] = np.nan
    df["avgy"] = np.nan
    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        fixMask = df.TYPE== "FIX"
        movementMask = df.BEFORE_MOVEMENT == False #Perché ho escluso la fase di planning?
        df.loc[trialMask & fixMask & movementMask, "avgx"] = df[trialMask & fixMask & movementMask].groupby(['LABEL_FIX'])["x"].transform(lambda row: row.mean())
        df.loc[trialMask & fixMask & movementMask, "avgy"] = df[trialMask & fixMask & movementMask].groupby(['LABEL_FIX'])["y"].transform(lambda row: row.mean())
        df.loc[trialMask & movementMask, "avgx"] = df.loc[trialMask & movementMask, "avgx"].fillna(method = "ffill")
        df.loc[trialMask & movementMask, "avgy"] = df.loc[trialMask & movementMask, "avgy"].fillna(method = "ffill")
        df.loc[trialMask & movementMask, "avgx"] = df.loc[trialMask & movementMask, "avgx"].fillna(method = "bfill")
        df.loc[trialMask & movementMask, "avgy"] = df.loc[trialMask & movementMask, "avgy"].fillna(method = "bfill")
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()





    #df = pd.read_csv("./OUTPUT/" + subjName +"dfSampleVis.csv", low_memory=False)




    ########### ALMOST ACTIVE NODES ###########
    print("     MODIFYING ACTIVE NODES WITH CLOSEST NODES INFORMATION PT:2")
    df["almostActiveNode"] = np.nan
    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        touchMask = df.TYPE == "TOUCH"

        backtrackMask = df.BACKTRACK == False
        #If the last node in almostActiveNode is different from the closestNode, then add it to the almostActiveNode
        df.loc[trialMask & touchMask & backtrackMask, "almostActiveNode"] = df.loc[trialMask & touchMask & backtrackMask, ["TOUCH_FIRST_CLOSEST_NODE", "activeNode"]].apply(
            lambda  x: str(Utilities.parse_string(x.activeNode) + Utilities.parse_string(x.TOUCH_FIRST_CLOSEST_NODE))
             if Utilities.parse_string(x.activeNode)[-1] != Utilities.parse_string(x.TOUCH_FIRST_CLOSEST_NODE)[0] 
             else x.activeNode, axis = 1)
        
        backtrackMask = df.BACKTRACK == True
        #If the first before the last node in almostActiveNode is equal to the closestNode, then remove last node from almostActiveNode
        df.loc[trialMask & touchMask & backtrackMask, "almostActiveNode"] = df.loc[trialMask & touchMask & backtrackMask, ["TOUCH_FIRST_CLOSEST_NODE", "activeNode"]].apply(
            lambda  x: str(Utilities.parse_string(x.activeNode)[:-1] )
             if Utilities.parse_string(x.activeNode)[-1] != Utilities.parse_string(x.TOUCH_FIRST_CLOSEST_NODE)[0]
             else x.activeNode, axis = 1)

    #Fillna with ffill
    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        df.loc[trialMask, "almostActiveNode"] = df.loc[trialMask, "almostActiveNode"].fillna(method='ffill')
        df.loc[trialMask, "almostActiveNode"] = df.loc[trialMask, "almostActiveNode"].fillna("NOTHING")
        #If MISSING_TOUCH is true then replace the almostActiveNode with the activeNode
        df.loc[trialMask & (df.MISSING_TOUCH == True), "almostActiveNode"] = df.loc[trialMask & (df.MISSING_TOUCH == True), "activeNode"]
        #Check if the difference between the two last nodes in almostActiveNode is not int actions2cardinal.keys(), then remove the last node from almostActiveNode
        df.loc[trialMask & (df.almostActiveNode != "NOTHING"), "almostActiveNode"] = df.loc[trialMask & (df.almostActiveNode != "NOTHING"), "almostActiveNode"].apply(lambda x: Utilities.parse_string(x)[:-1] if Utilities.checkMove(Utilities.parse_string(x), actions2cardinal) == False else x)
    #Convert to strings almostActiveNode
    df["almostActiveNode"] = df["almostActiveNode"].astype(str)
    #Replace [] with "NOTHING"
    df["almostActiveNode"] = df["almostActiveNode"].apply(lambda x: "NOTHING" if x == "[]" else x)

    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()
    
    

    #Convert to strings activeNode and almostActiveNode
    df["activeNode"] = df["activeNode"].astype(str)
    
    #Arrivati a questo punto activeNode e almostActiveNode usano solo NOTHING come segnale nullo 
   
    ########### ACTIVE NODES AT START AND END OF FIXATION ###########
    print("     ACTIVE NODES AT START AND END OF FIXATION")

    df["activeNodeStart"] = np.nan
    df["activeNodeEnd"] = np.nan

    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        #Find the first row of each LABEL_FIX
        firstTimeIndex = df[trialMask].groupby(['LABEL_FIX'])["t"].idxmin()
        #Find the corresponding activeNode
        activeNodeStart = df.loc[firstTimeIndex, "almostActiveNode"] ### HERE replace with almostActiveNode actiNode
        #Find the last row of each LABEL_FIX
        lastTimeIndex = df[trialMask].groupby(['LABEL_FIX'])["t"].idxmax()
        #Find the corresponding activeNode
        activeNodeEnd = df.loc[lastTimeIndex, "activeNode"]   ### HERE replace with almostActiveNode actiNode
        df.loc[firstTimeIndex, "activeNodeStart"] = activeNodeStart.values
        df.loc[firstTimeIndex, "activeNodeEnd"] = activeNodeEnd.values

        #In the trialMask indexes fill the nans with ffill
        filled_activeNodeStart = df.loc[trialMask].groupby(["LABEL_FIX"]).apply(lambda x: x.ffill().bfill())["activeNodeStart"]
        filled_activeNodeEnd = df.loc[trialMask].groupby(["LABEL_FIX"]).apply(lambda x: x.ffill().bfill())["activeNodeEnd"]
        df.loc[trialMask, "activeNodeStart"] = filled_activeNodeEnd.reset_index(level=0, drop=True)
        df.loc[trialMask, "activeNodeEnd"] = filled_activeNodeEnd.reset_index(level=0, drop=True)
        #Replace the remaining nans with the "NOTHING"
        df.loc[trialMask, "activeNodeStart"] = df.loc[trialMask, "activeNodeStart"].fillna("NOTHING")
        df.loc[trialMask, "activeNodeEnd"] = df.loc[trialMask, "activeNodeEnd"].fillna("NOTHING")
        """
        df.loc[trialMask, "activeNodeStart"] = df.loc[trialMask, "activeNodeStart"].fillna(method = "ffill")
        df.loc[trialMask, "activeNodeEnd"] = df.loc[trialMask, "activeNodeEnd"].fillna(method = "ffill")
        #Replace the remaining nans with the "NOTHING"
        df.loc[trialMask, "activeNodeStart"] = df.loc[trialMask, "activeNodeStart"].fillna("NOTHING")
        df.loc[trialMask, "activeNodeEnd"] = df.loc[trialMask, "activeNodeEnd"].fillna("NOTHING")"""
    
    #Convert to strings activeNodeStart and activeNodeEnd
    df["activeNodeStart"] = df["activeNodeStart"].astype(str)
    df["activeNodeEnd"] = df["activeNodeEnd"].astype(str)

  

    df.to_csv(("./OUTPUT/" + subjName +"dfSampleVis.csv"), index=False)
    #df = pd.read_csv("./OUTPUT/" + subjName +"dfSampleVis.csv", low_memory=False)
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()
    

    #Check
    #For each row with BACKTRACK == True check if the last tuple in Utilities.parse_string(df.at[i, "activeNodeStart"]) is in Utilities.parse_string(df.at[i, "activeNodeEnd"]). If not remove it from activeNodeStart
    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        for i in df[trialMask].index:
            startTuple = Utilities.parse_string(df.at[i, "activeNodeStart"])
            endTuple = Utilities.parse_string(df.at[i, "activeNodeEnd"])
            if startTuple != [] and endTuple != []:
                if Utilities.parse_string(df.at[i, "activeNodeStart"])[-1] not in Utilities.parse_string(df.at[i, "activeNodeEnd"]):
                    df.at[i, "activeNodeStart"] = Utilities.parse_string(df.at[i, "activeNodeStart"])[:-1]  
                    #Converto to str
                    df.at[i, "activeNodeStart"] = str(df.at[i, "activeNodeStart"])
    #Replace all the "[]" with "NOTHING"
    df["activeNodeStart"] = df["activeNodeStart"].apply(lambda x: "NOTHING" if x == "[]" else x)
    df["activeNodeEnd"] = df["activeNodeEnd"].apply(lambda x: "NOTHING" if x == "[]" else x)




    #A questo punto tutti i sensori di nodi sono stati convertiti in stringhe e tutti i nodi vuoti sono stati convertiti in "NOTHING"


    ########### MOTIF ###########
    print("     FIND THE MOTIVES")
    df["motif"] = pd.Series(dtype = "object")
    df["relMotif"] = pd.Series(dtype = "object")
    df["angle"] = np.nan
    for TRIAL_INDEX in range(94):
        print("          TRIAL_INDEX:", TRIAL_INDEX)
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX

        #Convert the activeNodeStart and activeNodeEnd to tuples by Utilities.parse_string if different from "NOTHING"
        df.loc[trialMask, "activeNodeStart"] = df.loc[trialMask,"activeNodeStart"].apply(lambda x: Utilities.parse_string(x))
        df.loc[trialMask, "activeNodeEnd"]  = df.loc[trialMask, "activeNodeEnd"].apply(lambda x: Utilities.parse_string(x))

        #Replace the ”[]" with "NOTHING"
        df.loc[trialMask, "activeNodeStart"] = df.loc[trialMask, "activeNodeStart"].apply(lambda x: "NOTHING" if x == [] else x)
        df.loc[trialMask, "activeNodeEnd"] = df.loc[trialMask, "activeNodeEnd"].apply(lambda x: "NOTHING" if x == [] else x)
        
        #print(df.loc[trialMask, "activeNodeStart"].unique())

        #Use the activeNodeStart and activeNodeEnd to find the corresponding motif nodes
        dfTrial = df[trialMask].reset_index()
        df.loc[trialMask, "motiveNodes"] = np.array([Utilities.find_missing_tuples([n for n in dfTrial.at[i, "activeNodeEnd"]], [n for n in dfTrial.at[i, 'activeNodeStart']]) for i in range(0, len(dfTrial))], dtype=object)
        dfTrial = df[trialMask].reset_index()
        #print(dfTrial["motiveNodes".unique()])
        for location in range(len(df[trialMask])):
            if len(dfTrial.at[location, 'motiveNodes']) == 1:
                #If there is only one node in the motive then there is no motif
                dfTrial.at[location, 'motif'] = "NOTHING"
                continue
            elif dfTrial.at[location, 'motiveNodes'] == ["NOTHING"]:
                #If there is no motive then there is no motif
                dfTrial.at[location, 'motif'] = "NOTHING"
                continue
            else:
                #If there is a motive then there is a motif
                dfTrial.at[location, 'motif'] = "".join(Utilities.convertIntoMotif(dfTrial.at[location, 'motiveNodes']))

        df.loc[trialMask, "motif"] = dfTrial["motif"].values
        df.loc[trialMask, "relMotif"] = df.loc[trialMask, "motif"].apply(Utilities.relativize).values
        df.loc[trialMask,"angle"] = df.loc[trialMask, "motif"].apply(lambda x: angleDict[x[0]] if type(x) == str else None)
        
        #Loop over the LABEL_FIX values in df[tiralMask]
        for LABEL_FIX in df[trialMask].LABEL_FIX.unique():
            labelFixMask = df[trialMask].LABEL_FIX == LABEL_FIX
            #Replace the "NOTHING" with np.nan
            df.loc[trialMask & labelFixMask, "motif"] = df.loc[trialMask & labelFixMask, "motif"].replace("NOTHING", np.nan)
            df.loc[trialMask & labelFixMask, "relMotif"] = df.loc[trialMask & labelFixMask, "relMotif"].replace("NOTHING", np.nan)
            df.loc[trialMask & labelFixMask, "motiveNodes"] = df.loc[trialMask & labelFixMask, "motiveNodes"].replace("NOTHING", np.nan)
            #Backfill the np.nan
            df.loc[trialMask & labelFixMask, "motif"] = df.loc[trialMask & labelFixMask, "motif"].bfill()
            df.loc[trialMask & labelFixMask, "relMotif"] = df.loc[trialMask & labelFixMask, "relMotif"].bfill()
            df.loc[trialMask & labelFixMask, "motiveNodes"] = df.loc[trialMask & labelFixMask, "motiveNodes"].bfill()
            #Replace the np.nan with "NOTHING"
            df.loc[trialMask & labelFixMask, "motif"] = df.loc[trialMask & labelFixMask, "motif"].replace(np.nan, "NOTHING")
            df.loc[trialMask & labelFixMask, "relMotif"] = df.loc[trialMask & labelFixMask, "relMotif"].replace(np.nan, "NOTHING")
            df.loc[trialMask & labelFixMask, "motiveNodes"] = df.loc[trialMask & labelFixMask, "motiveNodes"].replace(np.nan, "NOTHING")
    
    print("tempo trascorso: ", time.time() - t1)

    df.to_csv(("./OUTPUT/" + subjName +"dfSampleVis.csv"), index=False)

    print("Tempo totale: ", time.time() - t0 , " s")





    """"    ########### ALMOST ACTIVE NODES ###########
    print("     MODIFYING ACTIVE NODES WITH CLOSEST NODES INFORMATION PT:2")
    df["almostActiveNode"] = np.nan
    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        touchMask = df.TYPE == "TOUCH"
        #If the last node in almostActiveNode is different from the closestNode, then add it to the almostActiveNode
        df.loc[trialMask & touchMask, "almostActiveNode"] = df.loc[trialMask & touchMask, ["TOUCH_FIRST_CLOSEST_NODE", "activeNode"]].apply(
            lambda  x: Utilities.parse_string(x.activeNode) + [x.TOUCH_FIRST_CLOSEST_NODE]
             if Utilities.parse_string(x.activeNode)[-1] != x.TOUCH_FIRST_CLOSEST_NODE 
             else x.activeNode, axis = 1)
    #Fillna with ffill
    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        df.loc[trialMask, "almostActiveNode"] = df.loc[trialMask, "almostActiveNode"].fillna(method='ffill')
        df.loc[trialMask, "almostActiveNode"] = df.loc[trialMask, "almostActiveNode"].fillna("NOTHING")
    print("tempo trascorso: ", time.time() - t1)
    t1 = time.time()"""

    #1
    """    #Create a mask to identify the NaN values within each TRIAL_INDEX group
    mask = dfTouch.groupby("TRIAL_INDEX")["activeNode"].transform(lambda x: x.isna().cumsum() > 0)
    # Forward fill NaN values within the same TRIAL_INDEX group using the mask
    dfTouch.loc[mask, "activeNode"] = dfTouch.groupby("TRIAL_INDEX")["activeNode"].ffill()
    # Replace all the NaN values in dfMerge.activeNode by "None"
    dfTouch.loc[mask, "activeNode"] = dfTouch.activeNode.fillna("NOTHING")
    #let's make dfActiveNodes.activeNode a string
    dfTouch.loc[mask, "activeNode"] = dfTouch.activeNode.astype(str)"""





    """
    
    df["PAUSE"] = np.nan
    df.loc[df.TYPE == "TOUCH","PAUSE"] = False
    df["LABEL"] = None
    print("     DETECTING PAUSES")
    for TRIAL_INDEX in range(94):
        trialMask = (df.TRIAL_INDEX == TRIAL_INDEX)
        movingMask = (df.TYPE == "TOUCH")
        dfTouch = df[trialMask & movingMask].reset_index(drop=False)
        #rename the index column as originalIndex
        dfTouch.rename(columns={'index': 'originalIndex'}, inplace=True)
        #Find consecutive rows where SLOW_MOTION is True
        groups, startRDF, group_label, endRDF = [], None, 0, None
        for indexRDF, is_true in enumerate(dfTouch['SLOW_MOTION']):
            #print("startRDF:", startRDF, "is_true:", is_true)
            if is_true:
                startRDF = indexRDF if startRDF is None else startRDF
                endRDF = indexRDF
                
            elif startRDF is not None:
                groups.append((startRDF, endRDF, group_label))
                startRDF, group_label = None, group_label + 1 if endRDF - startRDF > planningIndexThreshold else group_label

        if startRDF is not None:
            groups.append((startRDF, endRDF, group_label))

        for start, end, label in groups:
            if end - start > planningIndexThreshold:
                dfTouch.loc[start:end, ['PAUSE', 'LABEL']] = True, label
                #Modify the original dataframe
                df.loc[dfTouch.loc[start:end, "originalIndex"], ['PAUSE', 'LABEL']] = True, label

    #Fillna with ffill  
    df["PAUSE"].fillna(method='ffill', inplace=True)
    """
    
    """
    print("     DETECTING FIXATIONS")
    fixationDurationThreshold = 80 #ms
    fixationIndexThreshold = int(fixationDurationThreshold / 5)
    fixationVelocityThreshold = 9
    df["FIXATION"] = np.nan
    df.loc[df.TYPE == "FIX","FIXATION"] = False
    df["LABEL_FIX"] = None
    df["FIXATION_DURATION"] = np.nan
    #print("     DETECTING FIXATIONS")
    for TRIAL_INDEX in range(94):
        trialMask = (df.TRIAL_INDEX == TRIAL_INDEX)
        movingMask = (df.TYPE == "FIX")
        dfFix = df[trialMask & movingMask].reset_index(drop=False)
        #Rename the index column as originalIndex
        dfFix.rename(columns={'index': 'originalIndex'}, inplace=True)
        #Find consecutive rows where eye velocity is under thrshold
        groups, startRDF, group_label, endRDF = [], None, 0, None
        for indexRDF, is_true in enumerate(dfFix.v < fixationVelocityThreshold):
            if is_true:
                startRDF = indexRDF if startRDF is None else startRDF
                endRDF = indexRDF
            elif startRDF is not None:
                groups.append((startRDF, endRDF, group_label))
                startRDF, group_label = None, group_label + 1 if endRDF - startRDF > fixationIndexThreshold else group_label
        if startRDF is not None:
            groups.append((startRDF, endRDF, group_label))
        for start, end, label in groups:
            if end - start > fixationIndexThreshold:
                dfFix.loc[start:end, ['FIXATION', 'LABEL_FIX']] = True, label
                df.loc[dfFix.loc[start:end, "originalIndex"], ['FIXATION', 'LABEL_FIX']] = True, label
                df.loc[dfFix.loc[start:end, "originalIndex"], ['FIXATION_DURATION']] = (dfFix.loc[end, "t"] - dfFix.loc[start, "t"])
            
    #Fillna with ffill  questa va corretta in modo che la propagazione avvenga solo tra le righe con lo stesso label_fix
    for TRIAL_INDEX in range(94):
        trialMask = df.TRIAL_INDEX == TRIAL_INDEX
        df.loc[trialMask, "FIXATION"] = df.loc[trialMask, "FIXATION"].fillna(method='ffill')
        df.loc[trialMask, "LABEL_FIX"] = df.loc[trialMask, "LABEL_FIX"].fillna(method='ffill')
        df.loc[trialMask, "FIXATION_DURATION"] = df.loc[trialMask, "FIXATION_DURATION"].fillna(method='ffill')
    print("tempo trascorso: ", time.time() - t1)
    """
