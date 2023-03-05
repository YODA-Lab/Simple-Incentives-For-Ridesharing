import numpy as np
from numpy.random import choice
import os
from copy import deepcopy
Dir = "conworld/"

#Create rectangluar "zones" connected at edges.
#Additive vs Subtractive approach
class GridWorld():
    """
    Simple 2D grid, where each element's neighbours are the elements adjacent to it.
    1 means node exists, 0 means it does not.
    Create map within world_h x world_w by either adding regions of 1's or creating regions of 0's
    """
    def __init__(self, world_w, world_h):
        self.world = [[1 for i in range(world_w)] for j in range(world_h)] 
        self.height = world_h
        self.width=world_w
        self.nodes = None
        self.zone_map = [["None" for i in range(world_w)] for j in range(world_h)]    
    
    def modify_block(self, x, y, w, h, create=True):
        #check if area is feasible
        assert x<self.width
        assert y<self.height
        #Make sure the area is within bounds
        if x+w > self.width: w = self.width-x
        if y+h > self.height: h = self.height-y
        
        val = 1 if create else 0
        for i in range(y, y+h):
            for j in range(x, x+w):
                self.world[i][j]=val
        
        #Since map was modified, de-assign nodes
        self.nodes = None
    
    def create_nodes(self):
        #Assign ID's to nodes that are '1'
        current_node = 0
        nodes = {}
        for i in range(self.height):
            for j in range(self.width):
                if self.world[i][j]==1:
                    a = {}
                    a['id'] = current_node
                    a['loc'] = (i,j)
                    a['adj'] = []
                    a['zone'] = self.zone_map[i][j]
                    a['subzone'] = None
                    adjacents = [(1,0),(0,1),(-1,0),(0,-1)]
                    for k,l in adjacents:
                        if (i+k,j+l) in nodes:
                            nodes[i+k,j+l]['adj'].append(a['id'])
                            a['adj'].append(nodes[i+k,j+l]['id'])
                    nodes[i,j] = a
                    current_node+=1
        self.nodes = nodes
        return nodes
        
    def assign_zone(self, name, x, y, w, h):
        ##Assigns zone name to the select area given by the rectangle (x,y,w,h)
        #check if area is feasible
        assert x<self.width
        assert y<self.height
        #Make sure the area is within bounds
        if x+w > self.width: w = self.width-x
        if y+h > self.height: h = self.height-y
        
        if not self.nodes:
            self.create_nodes()
        for i in range(y, y+h):
            for j in range(x, x+w):
                self.nodes[i,j]['zone']=name
                self.zone_map[i][j]=name
        

    def check_connectivity(self):
        return False

    def get_connectivity_graph(self):
        if not self.nodes:
            self.create_nodes()
        graph = [[] for i in range(len(self.nodes))]
        for node in self.nodes.values():
            graph[node['id']] = node['adj']
        return graph

    def get_nodes(self):
        if not self.nodes:
            self.create_nodes()
        nodes = [[] for i in range(len(self.nodes))]
        for node in self.nodes.values():
            nodes[node['id']] = node
        return nodes
    
    def get_zone_maps(self):
        # Returns list of node id's for each zone in a dict
        if not self.nodes:
            self.create_nodes()

        zones = {}
        for node in self.nodes.values():
            if node['zone'] in zones.keys():
                zones[node['zone']].append(node['id'])
            else:
                zones[node['zone']] = [node['id']]
        return zones

    def __repr__(self):
        retstr = ''
        if not self.nodes:
            self.create_nodes()
        for i,row in enumerate(self.world):
            txt = ''
            for j,item in enumerate(row):
                txt+=str(self.nodes[i,j]["zone"][-1]) + ' ' if item else '  '
                # txt+=str(item) + ' ' if item else '  '
            retstr+=txt+'\n'
        return retstr

def BFS(graph, node):
    #returns shortest distances to each node from given node
    distances = [() for _ in range(len(graph))]
    distances[node] = (node,0, node)  #current node, distance from source, previous node in path
    visited = [node]
    queue = [(node,0, node)]
    
    while queue:
        current_node, depth, ancestor_neighbour = queue.pop(0) #ancestor_neighbour keeps track of which node was the firststep from source
        for neighbour in graph[current_node]:
            if neighbour not in visited:
                if depth==0:
                    ancestor = neighbour
                else:
                    ancestor = ancestor_neighbour
                distances[neighbour] = (neighbour, depth +1, ancestor)
                queue.append((neighbour, depth+1, ancestor))
                visited.append(neighbour)
    return distances

#Simple 2 area map
# world = GridWorld(10,10)
# world.modify_block(0,5,5,5,False)
# world.modify_block(6,0,4,5,False)
# world.assign_zone("zone1",0,0,6,5)
# world.assign_zone("zone2",5,5,5,5)
# zone_weights = {}  #zone_name: (source_probability, dest_probability)
# zone_weights["zone1"] = (0.5, 0.5)
# zone_weights["zone2"] = (0.5, 0.5)

# 4 quadrants map
sz = 15
mid_sz = 5
world = GridWorld(sz,sz)
world.assign_zone("zone1",0,0,sz//2,sz//2)
world.assign_zone("zone2",sz//2,0,sz-sz//2,sz//2)
world.assign_zone("zone3",0,sz//2,sz//2,sz - sz//2)
world.assign_zone("zone4",sz//2,sz//2,sz-sz//2,sz-sz//2)
world.assign_zone("zone5",sz//2-mid_sz//2,sz//2-mid_sz//2,mid_sz,mid_sz)

world.modify_block(0,sz//2,sz,1,False)
world.modify_block(sz//2,0,1,sz,False)
world.modify_block(sz//2-mid_sz//2,sz//2-mid_sz//2,mid_sz,mid_sz,True)

# world = GridWorld(11,11)
# world.assign_zone("zone1",0,0,5,5)
# world.assign_zone("zone2",5,0,6,5)
# world.assign_zone("zone3",0,5,5,6)
# world.assign_zone("zone4",5,5,6,6)
# world.assign_zone("zone5",4,4,3,3)
# 
# world.modify_block(0,5,11,1,False)
# world.modify_block(5,0,1,11,False)
# world.modify_block(4,4,3,3,True)

zone_weights = {}  #zone_name: (source_probability, dest_probability)
zone_weights["zone1"] = (0.5, 0.2) #Lots of rides coming from here, very few going out 
zone_weights["zone2"] = (0.1, 0.3) #Very likely to land here, very unlikely to get out
zone_weights["zone3"] = (0.2, 0.2)
zone_weights["zone4"] = (0.15, 0.2)
zone_weights["zone5"] = (0.05, 0.1)

print(world)

graph = world.get_connectivity_graph()
num_nodes = len(graph)
data = [BFS(graph,i) for i in range(num_nodes)]

#Get traveltime
traveltime = [[data[i][j][1]*60.0 for j in range(num_nodes)] for i in range(num_nodes)]
# Write to a file
OUTFILE = Dir + 'zone_traveltime.csv'
np.savetxt(OUTFILE, traveltime, delimiter=',', fmt='%.1f')

#Get zone_dist
#Distances = Time in mins for now
# Write to a file
OUTFILE = Dir + 'zone_dist.csv'
np.savetxt(OUTFILE, traveltime, delimiter=',', fmt='%.1f')

#Get zone_path
path_data = [[data[i][j][2] for j in range(num_nodes)] for i in range(num_nodes)]
# Write to a file
OUTFILE = Dir + 'zone_path.csv'
np.savetxt(OUTFILE, path_data, delimiter=',', fmt='%d')

#Get ignored zones
ignored_nodes = [9999]
# Write to a file
OUTFILE = Dir + 'ignorezonelist.txt'
np.savetxt(OUTFILE, ignored_nodes, delimiter=',', fmt='%d')

#Set deterministic taxi initial states
taxi_locs = [np.random.randint(num_nodes) for _ in range(3000)]
# Write to a file
OUTFILE = Dir + 'taxi_3000_final.txt'
np.savetxt(OUTFILE, taxi_locs, delimiter='\n', fmt='%d')

import json, csv
node_info = world.get_nodes()

#GENERATE LATLONG
min_x, max_x = (0,1)
min_y, max_y = (0,1)
largest_x = 0
largest_y = 0
#For gridworld, assuming it starts at (0,0)
for node in node_info:
    x,y = node['loc']
    if x>largest_x: largest_x = x
    if y>largest_y: largest_y = y

header = ['id','lon','lat']
OUTFILE = Dir + 'zone_latlong.csv'
with open(OUTFILE, mode='w', newline='') as latlong_file:
    latlong_writer = csv.writer(latlong_file, delimiter=',')
    latlong_writer.writerow(header)
    for i, node in enumerate(node_info):
        x,y = node['loc']
        node_id = int(node['id'])
        lon = (x - 0)/(largest_x-0) * (max_x - min_x) + min_x
        lat = (y - 0)/(largest_y-0) * (max_y - min_y) + min_y
        node_info[i]['lon'] = lon
        node_info[i]['lat'] = lat
        latlong_writer.writerow([node_id, lon, lat])
        
#Write node_info
OUTFILE = Dir + 'neighbourhood_zones.json'
json.dump(node_info, open(OUTFILE, 'w'), sort_keys=True, indent=4)

zone_list = list(set([pt['zone'] for pt in node_info]))
zone_nodes = {zone:[] for zone in zone_list}
for pt in node_info:
    zone_nodes[pt["zone"]].append(pt)
subzone_list = list(set([pt['subzone'] for pt in node_info]))
zone_subzones = {zone:[] for zone in zone_list}
OUTFILE = Dir + 'neigh_zipcodes.json'
json.dump(zone_subzones, open(OUTFILE, 'w'))

#Generate Roads:
roads = []
title = ['x1','y1','x2','y2']
for p in range(len(node_info)):
    for k in node_info[p]["adj"]:
        roads.append({"x1":node_info[p]["lon"], "y1":node_info[p]["lat"],
                            "x2":node_info[k]["lon"], "y2":node_info[k]["lat"]})
with open(Dir + 'roads.csv', 'w', newline ='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=title)
    writer.writeheader()
    for i in roads:
        writer.writerow(i)

#Generate centroids
subzone_centroids = {}
zone_centroids = {}
for zone in zone_list:
    lon=[]
    lat=[]
    for pt in zone_nodes[zone]:
        lon.append(pt['lon'])
        lat.append(pt['lat'])
    zone_centroids[zone] = [sum(lon)/len(lon), sum(lat)/len(lat)]
centroids = {"subzone":subzone_centroids, "zone": zone_centroids}
json.dump(centroids, open( Dir+"centroids.json", 'w' ))

exit()
##REQUESTS
zone_info = world.get_zone_maps()
zones = list(zone_info.keys())
#Spatial distribution
zone_src_weights = [zone_weights[key][0] for key in zones]
zone_dest_weights = [zone_weights[key][1] for key in zones]

#Temporal distribution??
"""
Have different zonal weights at different times of the day
"""
#generate requests per min
days = range(1,21)
for day in days:
    print("Day: ",day)
    mean_rpm = 10 # mean
    std_rpm = 3 #St.Dev
    time_steps = 1440
    req_file = str(time_steps) +'\n'
    for step in range(time_steps):
        req_file+= "Flows:{}-{}".format(step,step) +'\n'
        num_samples = int(np.random.normal(mean_rpm, std_rpm))
        if num_samples<0: num_samples = 1
        for i in range(num_samples):
            from_zone = choice(zones, 1, p=zone_src_weights)[0]
            to_zone = choice(zones, 1, p=zone_dest_weights)[0]
            from_loc = choice(zone_info[from_zone], 1)[0]
            to_loc = choice(zone_info[to_zone], 1)[0]
            # from_loc = np.random.randint(num_nodes)
            # to_loc = np.random.randint(num_nodes)
            if from_loc!=to_loc:
                req_file+= str(from_loc)+','+str(to_loc)+',1.0\n'
    OUTFILE = Dir + 'files_60sec/' + 'test_flow_5000_{}.txt'.format(day)
    with open(OUTFILE, "w") as outfile:
        outfile.write(req_file)