import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Any, Optional


def _calculate_distance(p1: np.ndarray, p2: np.ndarray, metric: str) -> float:
    if metric == 'euclidean':
        return np.linalg.norm(p1 - p2)
    elif metric == 'cosine':
        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        if norm_p1 == 0 or norm_p2 == 0:
            return 1.0 
        
        sim = np.dot(p1, p2) / (norm_p1 * norm_p2)
        return 1.0 - sim
    else:
        raise ValueError("Metric must be 'euclidean' or 'cosine'")
    
def _jaccard_similarity(set1: set, set2: set) -> float:
    intersection_size = len(set1.intersection(set2))
    union_size = len(set1.union(set2))
    return intersection_size / union_size if union_size > 0 else 0
    

def calculate_evolutionary_graph(
    time_series_centers: List[np.ndarray],
    metric: str = 'cosine'
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[Tuple[Any, Any, float, str]]]:

    all_points = {}
    for t, clusters in enumerate(time_series_centers):
        for i, pos in enumerate(clusters):
            all_points[(t, i)] = pos

    final_edges = []
    num_timesteps = len(time_series_centers)

    for t in range(num_timesteps - 1):
        parents = time_series_centers[t]
        children = time_series_centers[t+1]

        if parents.size == 0 or children.size == 0:
            continue

        num_parents = len(parents)
        num_children = len(children)

        dist_matrix = np.zeros((num_parents, num_children))
        for i in range(num_parents):
            for j in range(num_children):
                dist_matrix[i, j] = _calculate_distance(parents[i], children[j], metric)

        best_child_for_parent = {i: (np.argmin(dist_matrix[i, :]), np.min(dist_matrix[i, :])) for i in range(num_parents)}
        best_parent_for_child = {j: (np.argmin(dist_matrix[:, j]), np.min(dist_matrix[:, j])) for j in range(num_children)}

        connected_parents = set()
        connected_children = set()
        
        for parent_idx, (child_idx, dist) in best_child_for_parent.items():
            if best_parent_for_child[child_idx][0] == parent_idx:
                u, v = (t, parent_idx), (t + 1, child_idx)
                final_edges.append((u, v, dist, 'backbone'))
                connected_parents.add(parent_idx)
                connected_children.add(child_idx)

        for child_idx in range(num_children):
            if child_idx not in connected_children:
                parent_idx, dist = best_parent_for_child[child_idx]
                u, v = (t, parent_idx), (t + 1, child_idx)
                final_edges.append((u, v, dist, 'rescue_child'))
                connected_parents.add(parent_idx)
                connected_children.add(child_idx) 

        for parent_idx in range(num_parents):
            if parent_idx not in connected_parents:
                child_idx, dist = best_child_for_parent[parent_idx]
                u, v = (t, parent_idx), (t + 1, child_idx)
                final_edges.append((u, v, dist, 'rescue_parent'))

    return all_points, final_edges

import networkx as nx
import itertools
def simplify_evolutionary_graph(
    all_points_map: Dict[Tuple[int, int], np.ndarray],
    populations_map: Dict[Tuple[int, int], int],
    edges: List[Tuple[Any, Any, float, str]],
    epsilon_merge: float = 0.5,
    theta_topo: float = 0.5,
    metric: str = 'cosine'
) -> Tuple[Dict[Tuple[int, int], np.ndarray], List[np.ndarray], List[Tuple[Any, Any, float, str]]]:
    """
    Simplifies an evolutionary graph by merging similar nodes and ensures final node IDs are integers.
    """
    # 1. Build NetworkX DiGraph
    G = nx.DiGraph()
    max_time = 0
    if all_points_map:
        max_time = max(t for t, i in all_points_map.keys())

    for node, centroid in all_points_map.items():
        G.add_node(node, time=node[0], centroid=centroid, population=populations_map.get(node, 1))

    for u, v, dist, edge_type in edges:
        G.add_edge(u, v, distance=dist, type=edge_type)

    times = sorted(list({data['time'] for _, data in G.nodes(data=True)}))
    
    # 2. Iteratively merge nodes at each time point
    merge_counter = 0
    for t in times:
        while True:
            merged_in_pass = False
            nodes_at_t = [n for n, data in G.nodes(data=True) if data['time'] == t]
            
            for u, v in itertools.combinations(nodes_at_t, 2):
                if u not in G or v not in G: continue
                dist = _calculate_distance(G.nodes[u]['centroid'], G.nodes[v]['centroid'], metric)
                if dist >= epsilon_merge: continue
                parents_u, parents_v = set(G.predecessors(u)), set(G.predecessors(v))
                children_u, children_v = set(G.successors(u)), set(G.successors(v))
                if _jaccard_similarity(parents_u, parents_v) < theta_topo or \
                   _jaccard_similarity(children_u, children_v) < theta_topo:
                    continue
                
                print(f"Merging nodes {u} and {v} at time {t}...")
                pop_u, pop_v = G.nodes[u]['population'], G.nodes[v]['population']
                total_pop = pop_u + pop_v
                new_centroid = (G.nodes[u]['centroid'] * pop_u + G.nodes[v]['centroid'] * pop_v) / total_pop
                new_node_id = (t, f"m{merge_counter}")
                merge_counter += 1
                G.add_node(new_node_id, time=t, centroid=new_centroid, population=total_pop)
                all_parents, all_children = parents_u.union(parents_v), children_u.union(children_v)
                for parent in all_parents:
                    new_dist = _calculate_distance(G.nodes[parent]['centroid'], new_centroid, metric)
                    types = [G.get_edge_data(parent, n)['type'] for n in [u, v] if G.has_edge(parent, n)]
                    new_type = 'rescue_parent' if 'rescue_parent' in types else 'rescue_child' if 'rescue_child' in types else 'backbone'
                    G.add_edge(parent, new_node_id, distance=new_dist, type=new_type)
                for child in all_children:
                    new_dist = _calculate_distance(new_centroid, G.nodes[child]['centroid'], metric)
                    types = [G.get_edge_data(n, child)['type'] for n in [u,v] if G.has_edge(n, child)]
                    new_type = 'rescue_parent' if 'rescue_parent' in types else 'rescue_child' if 'rescue_child' in types else 'backbone'
                    G.add_edge(new_node_id, child, distance=new_dist, type=new_type)
                G.remove_nodes_from([u, v])
                merged_in_pass = True
                break
            if not merged_in_pass: break

    # 3. Re-index nodes to have continuous integer IDs
    print("Re-indexing nodes to continuous integers...")
    relabel_map = {}
    for t in times:
        nodes_at_t = [n for n, data in G.nodes(data=True) if data['time'] == t]
        sorted_nodes = sorted(nodes_at_t, key=lambda n: G.nodes[n]['centroid'][0])
        for i, old_node_id in enumerate(sorted_nodes):
            relabel_map[old_node_id] = (t, i)
    
    # 【FIX】 The error occurs here. By removing copy=False (or setting copy=True),
    # we allow networkx to create a new graph to resolve naming conflicts.
    G = nx.relabel_nodes(G, relabel_map, copy=True)

    # 4. Convert NetworkX graph back to output format
    new_all_points_map = {n: data['centroid'] for n, data in G.nodes(data=True)}
    new_edges = [(u, v, data['distance'], data['type']) for u, v, data in G.edges(data=True)]
    
    # 【FIX】 Simplified and more robust logic for creating the new_centers list
    new_centers = []
    for t in range(max_time + 1):
        # Collect centroids for the current time step
        centroids_at_t = [
            data['centroid'] for n, data in G.nodes(data=True)
            if data['time'] == t
        ]
        # Ensure the list is sorted by the new integer index
        if centroids_at_t:
            # Get the nodes and sort them by their integer index
            nodes_at_t = sorted([n for n in G.nodes() if n[0] == t], key=lambda n: n[1])
            # Get centroids in the correct order
            ordered_centroids = [G.nodes[n]['centroid'] for n in nodes_at_t]
            new_centers.append(np.array(ordered_centroids))
        else:
            new_centers.append(np.array([]))

    print("Graph simplification and re-indexing complete.")
    return new_all_points_map, new_centers, new_edges


