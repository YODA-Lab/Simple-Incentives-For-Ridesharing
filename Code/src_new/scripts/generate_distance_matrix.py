"""
This file estimates the distance between any 2 intersections and saves it in a file
It does this by taking 2 inputs: (1) shortest path between 2 locations, (2) lat-long for locations
It then finds the distance between any 2 intersections as follows:
    (1) It decomposes the distance between 2 intersections as a sum over the lengths of the contiguous road segments in the shortest path between these 2 intersections
    (2) For each contiguous road segment, it assumes that its length is equal to the straight-line distance between the two intersections that bookend it
"""

from pandas import read_csv
from math import radians, cos, sin, asin, sqrt
import numpy as np

# For debugging
import pdb
pdb.set_trace()

# Initialise Constants
DATA_DIR = '../../data/ny/'

# INPUTS
# File with shortest path between any 2 intersections
# The shortest path is stored as a matrix with the (x, y) entry denoting the next intersection on the shortest path between x and y
SHORTESTPATH_FILE: str = DATA_DIR + 'zone_path.csv'
shortest_path = read_csv(SHORTESTPATH_FILE, header=None).values
#   Sanity Check
assert len(shortest_path.shape) == 2
assert shortest_path.shape[0] == shortest_path.shape[1]

# File with lat-long for each intersection
LATLONG_FILE: str = DATA_DIR + 'zone_latlong.csv'
lat_longs = read_csv(LATLONG_FILE, header=None, index_col=0).values
#   Sanity Check
num_locs = len(lat_longs)
assert num_locs == shortest_path.shape[0]

# File with a list of locations to ignore
IGNOREDZONES_FILE: str = DATA_DIR + 'ignorezonelist.txt'
ignored_zones = read_csv(IGNOREDZONES_FILE, header=None).values.flatten()
active_locs = [i for i in range(num_locs) if i not in ignored_zones]


# PROCESSING
# Helper functions
#   Returns the straight-line distance between 2 locations
def haversine(src, dest):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # Get lat-long for src and dest
    lon1, lat1 = lat_longs[src]
    lon2, lat2 = lat_longs[dest]

    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * asin(sqrt(a))
    mi = 3958.8 * c  # 3958.8 is the radius of the earth in miles

    return mi


#   Populates the shortest path between src and dest recursively
#       ASSUMPTION: shortest_path and distances are global vars
def populate_dist(src, dest):
    # Traverse along the shortest path until you find a loc for which distances[loc, dest] is known
    #   Note: In the worst case, it will traverse till dest because distances[dest, dest] is set to 0.
    cur_loc = src
    path = []
    while distances[cur_loc][dest] == -1:
        path.append(cur_loc)
        cur_loc = shortest_path[cur_loc][dest]

    # For all intermediate locs from src to cur_loc
    next_loc = cur_loc
    while path:
        # Update distances[loc, dest]
        loc = path.pop()

        if distances[loc, next_loc] == -1:
            distances[loc, next_loc] = haversine(loc, next_loc)
        dist_to_next = distances[loc, next_loc]
        dist_after_next = distances[next_loc, dest]
        distances[loc, dest] = dist_to_next + dist_after_next

        next_loc = loc


# Initialise Variable
#   The distances between any two intersections are stored in a matrix
#   The (x, y) entry in the matrix is the distance between 2 locations x and y
#   ASSUMPTION: Distances are always positive, so any negative distance implies that the distance has not been populated so far
distances = np.zeros((num_locs, num_locs)) - 1

# Populate Distances
#   Base case for recursion: set distance of location to itself to 0
for i in active_locs:
    distances[i, i] = 0

#   For each (x, y) pair of intersections, find the distance
for src in active_locs:
    print(src)
    for dest in active_locs:
        if distances[src, dest] == -1:
            populate_dist(src, dest)

# Sanity check
for src in ignored_zones:
    for dest in ignored_zones:
        assert distances[src, dest] == -1

# Write to a file
OUTFILE: str = DATA_DIR + 'zone_dist.csv'
np.savetxt(OUTFILE, distances, delimiter=',', fmt='%.4f')
