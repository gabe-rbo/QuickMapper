import random
import networkx as nx
import matplotlib.pyplot as plt
import math
import plotly.graph_objects as go
from scipy.spatial.distance import pdist, squareform
import numpy as np

def quick_mapper(G_raw, max_loops=1, min_modularity_gain=0.000001):
    """The fast relabeling algorithm from the paper."""
    V = G_raw['V']
    E = G_raw['E']
    
    adj = {v: set() for v in V}
    for u, v in E:
        adj[u].add(v)
        adj[v].add(u)
    
    m = len(E)
    
    L = {v: v for v in V}
    
    # Edge case for an empty graph
    if m == 0: 
        return {'V': list(set(L.values())), 'E': []}, L
        
    degree = {v: len(adj[v]) for v in V}
    
    def get_P(i, j):
        return (degree[i] * degree[j]) / (2.0 * m)
    
    num_of_loops = 0
    modularity_gain = 1000.0
    
    while modularity_gain > min_modularity_gain and num_of_loops < max_loops:
        vertex_order = list(V)
        random.shuffle(vertex_order)
        
        for vertex in vertex_order:
            neighbors = adj[vertex]
            if not neighbors: continue
            
            NbrLabelSet_vertex = {L[vertex]}
            for nbr in neighbors:
                NbrLabelSet_vertex.add(L[nbr])
            
            best_labels = []
            max_contribution = -float('inf')
            
            for label in NbrLabelSet_vertex:
                # Simplified local modularity gain calculation
                contribution = sum((1 - get_P(vertex, j)) for j in neighbors if L[j] == label)
                
                if contribution > max_contribution:
                    max_contribution = contribution
                    best_labels = [label]
                elif contribution == max_contribution:
                    best_labels.append(label)
                    
            L[vertex] = random.choice(best_labels)
            
        modularity_gain = 0 # Force exit after set passes for this simple demonstration
        num_of_loops += 1
        
    E_simple = set()
    for vertex in V:
        for nbr in adj[vertex]:
            if L[vertex] != L[nbr]:
                E_simple.add(tuple(sorted([L[vertex], L[nbr]])))
                
    G_simple = {'V': list(set(L.values())), 'E': list(E_simple)}
    
    # Return both the simplified graph AND the dictionary mapping original points to clusters
    return G_simple, L 

def build_epsilon_network(points, epsilon=0.2):
    V = list(range(len(points)))
    E = []
    
    # Calculate Euclidean distance between all pairs
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            dist = math.hypot(points[i][0] - points[j][0], points[i][1] - points[j][1])
            if dist <= epsilon:
                E.append((i, j))
                
    return {'V': V, 'E': E}

def plot_quick_mapper(points, G_simple, L):
    """
    Calculates the 3D centroids of the clusters and plots an interactive 
    3D network using Plotly.
    """
    # Calculate the 3D centroid for each simplified node
    centroids = {}
    for node_id in G_simple['V']:
        # Get all original points that were assigned to this cluster label
        cluster_indices = [i for i, label in L.items() if label == node_id]
        cluster_points = points[cluster_indices]
        # Average their x, y, z coordinates
        centroids[node_id] = np.mean(cluster_points, axis=0)

    # Extract coordinates for the nodes
    node_x = [centroids[node][0] for node in G_simple['V']]
    node_y = [centroids[node][1] for node in G_simple['V']]
    node_z = [centroids[node][2] for node in G_simple['V']]

    # Extract coordinates for the edges (inserting None to break the lines for Plotly)
    edge_x, edge_y, edge_z = [], [], []
    for u, v in G_simple['E']:
        edge_x.extend([centroids[u][0], centroids[v][0], None])
        edge_y.extend([centroids[u][1], centroids[v][1], None])
        edge_z.extend([centroids[u][2], centroids[v][2], None])

    # Create Plotly Traces
    trace_edges = go.Scatter3d(
        x=edge_x, y=edge_y, z=edge_z,
        mode='lines',
        line=dict(color='salmon', width=4),
        hoverinfo='none',
        name='Topological Connections'
    )

    trace_nodes = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers',
        marker=dict(
            size=10,
            color=node_z, # Color by Z-axis depth for visual pop
            colorscale='Viridis',
            line=dict(color='black', width=2)
        ),
        text=[f"Cluster {n}" for n in G_simple['V']],
        hoverinfo='text',
        name='Simplified Nodes'
    )

    trace_original = go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=2, color='lightblue', opacity=0.2),
        hoverinfo='none',
        name='Original Data'
    )

    # Assemble and Show
    fig = go.Figure(data=[trace_original, trace_edges, trace_nodes])
    fig.update_layout(
        title="Interactive 3D Quick Mapper Topology (Hollow Sphere)",
        scene=dict(
            xaxis=dict(showbackground=False),
            yaxis=dict(showbackground=False),
            zaxis=dict(showbackground=False)
        ),
        margin=dict(l=0, r=0, b=0, t=40),
        showlegend=True
    )
    fig.show()
