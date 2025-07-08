import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import re
import json
import networkx as nx
import math

def mapExp2Screen(mapName, x, y):

    level1Origin = [ 284, 1920 - 1215]
    level2Origin = [ 217, 1920 - 1282]
    level3Origin = [ 217, 1920 - 1415]
    linkEstimatedLenght = 132.65217391304347

    if mapName[1] == "5":
        x = (x/5. + 2  )*linkEstimatedLenght + level1Origin[0]
        y = (y/5. + 2  )*linkEstimatedLenght + level1Origin[1]
    elif int(mapName[3]) < 8:
        x = (x/5. + 2.5)*linkEstimatedLenght + level2Origin[0]
        y = (y/5. + 2.5)*linkEstimatedLenght + level2Origin[1]
    else: 
        x = (x/5. + 2.5)*linkEstimatedLenght + level3Origin[0]
        y = (y/5. + 3.5)*linkEstimatedLenght + level3Origin[1]
    return list(zip(list(x), list(y)))

def mapGraph2Screen(mapName, coords):
    level1Origin = [ 284, 1920 - 1215]
    level2Origin = [ 217, 1920 - 1282]
    level3Origin = [ 217, 1920 - 1415]
    linkEstimatedLenght = 132.65217391304347

    if len(coords) == 1:
        x, y = coords[0][0], coords[0][1]
    else:
        x = np.array([c[0] for c in coords])
        y = np.array([c[1] for c in coords])
    if mapName[1] == "5":
        x = x*linkEstimatedLenght + level1Origin[0]
        y = y*linkEstimatedLenght + level1Origin[1]
    elif int(mapName[3]) < 8:
        x = x*linkEstimatedLenght + level2Origin[0]
        y = y*linkEstimatedLenght + level2Origin[1]
    else:
        x = x*linkEstimatedLenght + level3Origin[0]
        y = y*linkEstimatedLenght + level3Origin[1]
    if len(coords) == 1:
        return x, y
    else:
        return list(zip(x,y))
    
def screenToMapGraph(mapName, coords):
    level1Origin = [284, 1920 - 1215]
    level2Origin = [217, 1920 - 1282]
    level3Origin = [217, 1920 - 1415]
    linkEstimatedLength = 132.65217391304347

    if len(coords) == 1:
        x, y = coords[0][0], coords[0][1]
    else:
        x = np.array([c[0] for c in coords])
        y = np.array([c[1] for c in coords])

    if mapName[1] == "5":
        x = (x - level1Origin[0]) / linkEstimatedLength
        y = (y - level1Origin[1]) / linkEstimatedLength
    elif int(mapName[3]) < 8:
        x = (x - level2Origin[0]) / linkEstimatedLength
        y = (y - level2Origin[1]) / linkEstimatedLength
    else:
        x = (x - level3Origin[0]) / linkEstimatedLength
        y = (y - level3Origin[1]) / linkEstimatedLength

    if len(coords) == 1:
        return x, y
    else:
        return list(zip(x, y))

    
def map_load(path):

    level1Origin = [ 284, 1920 - 1215]
    level2Origin = [ 217, 1920 - 1282]
    level3Origin = [ 217, 1920 - 1415]
    linkEstimatedLenght = 132.65217391304347

    mapName = path.split("/")[-1].split(".")[0]
    with open(path, 'r') as f:
        data = json.load(f)
    nodes = {}
    links = {}
    for n in data.get("nodes"):
        nodes[n.get("ID")] = n
    for l in data.get("links"):
        links[l.get("SOURCE"), l.get("TARGET")] = 1
    nodes_dict = {key:(nodes[key]["X"], nodes[key]["Y"]) for key in nodes.keys()}
    node_list = list(nodes_dict.values())
    link_list = [(nodes_dict[n[0]], nodes_dict[n[1]]) for n in links.keys()]
    
    g = nx.Graph()
    g.add_nodes_from(node_list)
    g.add_edges_from(link_list)
    
    col = {nodes_dict[i]:r["reward"] for i,r in enumerate(nodes.values())}
    
    nx.set_node_attributes(g, col, "reward")
    
    #identify the start
    start = list(g.nodes())[data["start"]]
    startAttribute = {node:0 if node != start else 1 for node in g.nodes() }
    nx.set_node_attributes(g, startAttribute, "start")

    #Let's set as attribute node pos in pixels
    if  mapName[1] == "5":
        pos={n:(n[0]*linkEstimatedLenght + level1Origin[0], n[1]*linkEstimatedLenght +level1Origin[1]) for n in g.nodes()}
    elif int(mapName[3]) < 8:
        pos={n:(n[0]*linkEstimatedLenght + level2Origin[0], n[1]*linkEstimatedLenght +level2Origin[1]) for n in g.nodes()}
    else:
        pos={n:(n[0]*linkEstimatedLenght + level3Origin[0], n[1]*linkEstimatedLenght +level3Origin[1]) for n in g.nodes()}

    nx.set_node_attributes(g, pos, "pos")

    return g

#function to convert a string to a list of tuples
def parse_string(input_string):
    # Extract the tuple strings using regular expression
    tuple_strings = re.findall(r'\((\d+), (\d+)\)', input_string)
    # Convert each tuple string into a tuple of integers
    result = [(int(x), int(y)) for x, y in tuple_strings]
    return result

def find_closest_nodes(input_pos, graph):

    g = graph
    pos = nx.get_node_attributes(g, 'pos')
    #Find the distance between the input position and all nodes
    distances = [np.linalg.norm(np.array(input_pos) - np.array(pos[node])) for node in g.nodes()]
    #Find the index of the four closest nodes and their distances
    closest_nodes = [list(g.nodes())[i] for i in np.argsort(distances)[:4]]
    closest_distances = [distances[i] for i in np.argsort(distances)[:4]]

    return closest_nodes, closest_distances

def split_into_consecutive_sublists(arr):
    sublists = []
    sublist = []
    
    for i, num in enumerate(arr):
        if i == 0:
            sublist.append(num)
        elif num == arr[i-1] + 1:
            sublist.append(num)
        else:
            sublists.append(sublist)
            sublist = [num]
    
    if sublist:
        sublists.append(sublist)
    
    return sublists

def rotateFloat(theta, coordinates):
        # Convert angle to radians
        theta_rad = math.radians(theta)
        # Compute sine and cosine of the angle
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        # Create the rotation matrix
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        # Convert coordinates to numpy array
        coordinates = np.array(coordinates)
        # Apply rotation to coordinates
        rotated_coordinates = np.dot(rotation_matrix, coordinates.T)
        #Convert into a list of tuples
        rotated_coordinates = [(x[0], x[1]) for x in rotated_coordinates.T.tolist()]
        return rotated_coordinates


class GraphPatternMatcher(object):
    def __init__(self, motif = None):
        self.moveDict = {"N": (0,1), "S": (0,-1), "E": (1,0), "W": (-1,0)}
        self.motif = self.def_motif(motif)
        self.pathMotif = []
    def rotate(self, theta, coordinates):
        # Convert angle to radians
        theta_rad = math.radians(theta)
        # Compute sine and cosine of the angle
        cos_theta = math.cos(theta_rad)
        sin_theta = math.sin(theta_rad)
        # Create the rotation matrix
        rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
        # Convert coordinates to numpy array
        coordinates = np.array(coordinates)
        # Apply rotation to coordinates
        rotated_coordinates = np.dot(rotation_matrix, coordinates.T)
        # Round and convert coordinates to integers
        rotated_coordinates = np.round(rotated_coordinates).astype(int)
        #Convert into a list of tuples
        rotated_coordinates = [(x[0], x[1]) for x in rotated_coordinates.T.tolist()]
        return rotated_coordinates

    def move(self, node, move):
        return tuple(np.array(node) + np.array(self.moveDict[move]))
    
    def def_motif(self, moveSequence):
        #Input motif: a list of "N, S, E, W" characters
        #Output motif: a list of coordinates
        #Let's encode the motif as a list of coordinates
        start = (0,0)
        motif = [start]
        for move in moveSequence:
            motif.append(self.move(motif[-1], move))
        return motif
    
    def find_motif(self, path = None, motif = None):
        pathMotif = []
        if motif is None:
            motif = self.motif
        #sliding window over the path
        for i in range(len(path) - len(motif) + 1):
            subpath = path[i:i+len(motif)]
            #Let's shift the subpath so that the first node is the origin
            subpathShifted = [(x[0] - subpath[0][0], x[1] - subpath[0][1]) for x in subpath]
            #rotate the motif of90, 180, 270 degrees and check if it matches the subpath
            for theta in [0, 90, 180, 270]:
                rotated_motif = self.rotate(theta, motif)
                if subpathShifted == rotated_motif:   
                    pathMotif.append((subpath, path[i], i, -theta))
        return pathMotif

def get_curvature(t, vx, vy):
    
    #VELOCITY
    dx_dt = vx
    dy_dt = vy
    velocity = [[dx_dt[ind], dy_dt[ind]] for ind, _ in enumerate(dx_dt)]
    ds_dt = np.sqrt(dx_dt * dx_dt + dy_dt * dy_dt)
    
    #TANGENTE
    tangent = np.array([1/ds_dt] * 2).transpose() * velocity
    tangent_x = tangent[:, 0]
    tangent_y = tangent[:, 1]
    deriv_tangent_x = np.gradient(tangent_x, t)
    deriv_tangent_y = np.gradient(tangent_y, t)
    dT_dt = np.array([ [deriv_tangent_x[i], deriv_tangent_y[i]] for i in range(deriv_tangent_x.size)])
    length_dT_dt = np.sqrt(deriv_tangent_x * deriv_tangent_x + deriv_tangent_y * deriv_tangent_y)
    
    #NORMALE
    normal = np.array([1/length_dT_dt] * 2).transpose() * dT_dt
    normal_x = normal[:, 0]
    normal_y = normal[:, 1]
    deriv_normal_x = np.gradient(normal_x, t)
    deriv_normal_y = np.gradient(normal_y, t)
    lenght_dN_dt = np.sqrt(deriv_normal_x * deriv_normal_x + deriv_normal_y * deriv_normal_y)
    
    #CURVATURA
    d2x_dt2 = np.gradient(dx_dt, t)
    d2y_dt2 = np.gradient(dy_dt, t)

    curvature = np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2) / (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    radius_of_curvature = 1/curvature
    
    #ANGULAR VELOCITY
    w_tangent = length_dT_dt
    w_norm    = lenght_dN_dt
    return curvature, radius_of_curvature, normal, tangent, w_tangent , w_norm

        
def visualize_map(mapName, x= None, y= None):
    level1Origin = [ 284, 1920 - 1215]
    level2Origin = [ 217, 1920 - 1282]
    level3Origin = [ 217, 1920 - 1415]
    linkEstimatedLenght = 132.65217391304347
    scale = 5
    g = map_load("/Users/Mattia/Desktop/EYE_TRACKER/MAPS/" + mapName[:-5])
    
    fig, ax = plt.subplots(figsize=(scale*2, scale*1920/1080))
    if mapName[1]=="5":
        pos={n:(n[0]*linkEstimatedLenght + level1Origin[0], n[1]*linkEstimatedLenght +level1Origin[1]) for n in g.nodes()}
    elif int(mapName[3]) < 8:
        pos={n:(n[0]*linkEstimatedLenght + level2Origin[0], n[1]*linkEstimatedLenght +level2Origin[1]) for n in g.nodes()}
    else:
        pos={n:(n[0]*linkEstimatedLenght + level3Origin[0], n[1]*linkEstimatedLenght +level3Origin[1]) for n in g.nodes()}

    #pos = nx.get_node_attributes(g, 'pos')
    reward = nx.get_node_attributes(g, 'reward')
    start = [n for n in g.nodes() if g.nodes[n]["start"] == 1][0]
    #color is grey if the node is not a reward, red otherwise. The start is black
    node_color = ["grey" if reward[n] == 0 else "red" for n in g.nodes()]
    node_color[list(g.nodes()).index(start)] = "black"
    nx.draw_networkx(g, pos, node_color = node_color, edge_color = "grey", width = 1.5,  node_size = 100, with_labels=False, ax = ax)
    #add a scatter at given coordinates
    if x is not None and y is not None:
        plt.scatter(x, y, color = "pink", s = 100)
    plt.axis('equal')
    plt.show()

def subpathIsInPath(subpath, path):
    #This function takes as input two list of elments annd checks if the first one is contained in the second one
    #If the first one is contained in the second one, the function returns the second list
    #Otherwise it returns False
    if len(subpath) > len(path):
        return False
    for i in range(len(path) - len(subpath) + 1):
        if path[i:i+len(subpath)] == subpath:
            return path
    return False

def sublistIsInList(list1, list2):
    #This function takes as input two list of elments annd checks if the first one is contained in the second one
    #If the first one is contained in the second one, the function returns the second list
    #Otherwise it returns False
    if len(list1) > len(list2):
        return False
    if all(item in list2 for item in list1):
        return list2
    return False

def nodesInAwarenessRadius(g, coords, r):
    #Receive as input a graph, an awareness radius and a point coordinates.
    #The output must be the nodes in the graph that are in the awareness radius of the point
    nodes = []
    for node in g.nodes():
        #cget "pos" attribute of the node
        node_coords = g.nodes[node]["pos"]
        #compute the distance between the node and the point
        distance = np.sqrt((node_coords[0] - coords[0])**2 + (node_coords[1] - coords[1])**2)
        #if the distance is less than the awareness radius, add the node to the list
        if distance <= r:
            nodes.append(node)
    return nodes


#Let's build the rotation matrix for a generic angle
def rotationMatrix(angle):
    angle = np.radians(angle)
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def rotateVector(vector, angle):
    #Return the rotated vector with int coordinates
    rotatedV = np.dot(rotationMatrix(angle), vector)
    return tuple(np.round(rotatedV).astype(int))

def convertIntoMotif(data):
    cardinalDict = {(0,1): "N", (0,-1): "S", (1,0): "E", (-1,0): "W"}
    #if data is a string, convert it into a list of tuple by Utilities.parse_string
    #if isinstance(data, str):
    if data == "NOTHING":
        return "NOTHING"
    #if len(data) > 1:
    #    data = parse_string(data)
    return [cardinalDict[tuple(n)] for n in np.diff(data, axis=0)]

def relativize(inputMotif):
    cardinal2actions = {"N": (0,1), "S": (0,-1), "E": (1,0), "W": (-1,0)}
    actions2cardinal = {(0,1): "N", (0,-1): "S", (1,0): "E", (-1,0): "W"}
    angleDict = {"N": 0, "E":90, "S":180, "W":270}
    if inputMotif == "NOTHING":
        return "NOTHING"
    else:
        angle = angleDict[inputMotif[0]]
        return "".join([actions2cardinal[rotateVector(cardinal2actions[card], angle)] for card in inputMotif])
    
def find_missing_tuples(current_tuple, previous_tuple):
    #if current_tuple == [] or previous_tuple == []:
    #    return current_tuple
    if current_tuple == ['N', 'O', 'T', 'H', 'I', 'N', 'G'] or previous_tuple == ['N', 'O', 'T', 'H', 'I', 'N', 'G']:
        return "NOTHING"

    if current_tuple == previous_tuple:
        return "NOTHING"

    #Find common tuples between current_tuple and previous_tuple
    common_tuples = []
    if len(current_tuple) > len(previous_tuple): 
        longestList = current_tuple 
        shortestList = previous_tuple
    else: 
        longestList = previous_tuple
        shortestList = current_tuple

    for tindex, t in enumerate(shortestList):
        if t == longestList[tindex]:
            common_tuples.append(t)
        else:
            break

    #newTuples will be the remaining part of current_tuple after the last common tuple
    newTuples = current_tuple[len(common_tuples):]
    #removedTuples will be the remaining part of previous_tuple after the last common tuple
    removedTuples = previous_tuple[len(common_tuples):]

    lastCommonTuple = common_tuples[-1]
    #reverse the list of removed tuples
    removedTuples.reverse()
    missing_tuple = removedTuples + [lastCommonTuple] + newTuples
    return missing_tuple

def checkMove(nodeLs, actions2cardinal):

    #Input: 
    # 1) a list of tuples nodeLs
    # 2) the dictionary from coordinate tuple to cardinal action

    #Main: 
    #check if the difference between the last two nodes is a key of the dictionary
    #return False otherwise
    if len(nodeLs) < 2:
        return False
    action = tuple(np.diff(nodeLs, axis=0)[-1])
    if action not in actions2cardinal.keys():
        return False
    return True



def get_classWithin(relativeMotif):
    if len(relativeMotif) == 1:
        return "SingularMotif"
    else:
        return relativeMotif[1]
    

def all_simple_paths_graph(G, source, targets, cutoff, depth):
    visited = dict.fromkeys([source])
    stack = [iter(G[source])]
    while stack:
        children = stack[-1]
        child = next(children, None)
        if child is None:
            stack.pop()
            visited.popitem()
        elif len(visited) < cutoff:
            if child in visited:
                continue
            depth_equality = len(targets & (set(visited.keys()) | {child})) == depth
            if depth_equality:#child in targets:
                yield list(visited) + [child]
            visited[child] = None
            depth_inequality = len(targets & set(visited.keys())) < depth+1
            if len(targets - set(visited.keys())) > len(targets)-depth:  # expand stack until find all targets
                stack.append(iter(G[child]))
            else:
                visited.popitem()  # maybe other ways to child
        else:  # len(visited) == cutoff:
            #(targets & (set(children) | {child})) - set(visited.keys()) tartget vicini
            #non visitati
            for target in (targets & (set(children) | {child})) - set(visited.keys()):
                    if len(targets & (set(visited.keys()) | {target}))==depth:
                        yield list(visited) + [target]
            stack.pop()
            visited.popitem()