"""
Author: Nicholas Glaze, Rice ECE (nkg2 at rice.edu)
Modified by mitch roddenberry for Michael T. Schaub

Code for converting ocean drifter data from jld2 format to array form.
"""

import h5py
import numpy as np
import networkx as nx
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
import os

def create_node_edge_incidence_matrix(elist):
    num_edges = len(elist)
    data = [-1] * num_edges + [1] * num_edges
    row_ind = [e[0] for e in elist] + [e[1] for e in elist]
    col_ind = [i for i in range(len(elist))] * 2
    B1 = csc_matrix(
        (np.array(data), (np.array(row_ind), np.array(col_ind))), dtype=np.int8)
    return B1.toarray()


def incidence_matrices(E, faces):
    # E - list of edges in order
    # faces - list of sorted faces in order
    # # edge_to_idx - dictionary converting 
    
    B1 = create_node_edge_incidence_matrix(E)
    B2 = np.zeros([len(E),len(faces)])
    
    for f_idx, face in enumerate(faces):
        face = sorted(face)
        f_edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [E.index(tuple(e)) for e in f_edges]
        
        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1

    return B1, B2

def paths_to_flows(E, paths):
    # assume each edge of E is sorted
    flows = []
    for p in paths:
        f = np.zeros(len(E))
        
        for idx in range(len(p)-1):
            i = p[idx]
            j = p[idx+1]
            flow_val = 1

            if i > j:
                r = j
                j = i
                i = r
                flow_val = -1
            elif i == j:
                continue

            e = (i,j)
            f[E.index(e)] += flow_val
            
        flows.append(f)
        
    return np.array(flows)
            
# graph Construction
#%%
f = h5py.File('dataBuoys.jld2', 'r')
f2 = h5py.File('dataBuoys-coords.jld2', 'r')

edge_list_np = (f['elist'][:] - 1).T # 1-index -> 0-index
edge_list = [tuple(e) for e in edge_list_np]

face_list_np = (f['tlist'][:] - 1).T # 1-index -> 0-index
face_list = [tuple(sorted(f)) for f in face_list_np]

G = nx.Graph()
G.add_edges_from(edge_list)

node_list = np.array(sorted(G.nodes))

B1, B2 = incidence_matrices(edge_list, face_list)

# trajectories

traj_nodes = [[f[x][()] - 1 for x in f[ref][()]]
              for ref in f['TrajectoriesNodes'][:]]

# convert trajectories to flows

flows = paths_to_flows(edge_list, traj_nodes) # each row = 1 flow


Hexcenters = [[x[0],x[1]] for x in f['HexcentersXY'][:]]

NodeID = [f2[ref][()]
              for ref in f2['HexToNodeIDs'][:]]



# Taxonomy:

# B1 : edge-node incidence matrix
# B2 : face-edge incidence matrix
# flows : matrix of flows, each row is a flow

# flows was built from the trajectories additively:
# if an edge appears twice going in the same direction
# flow has value +2
# if an edge goes forward and then backward
# flow has value 0
