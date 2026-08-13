import torch
import numpy as np
from typing import List, Dict, Tuple, Any, Union

def calibrated_generation_with_debug(
    initial_points: torch.Tensor,
    observation_times: List[float],
    sde_solver: callable,
    timepoint_lists_for_each_stage: List[List],
    v_m: Any,
    scale_m: Any,
    delta_t: float,
    eps: Union[float, callable],
    cluster_models: Dict[float, callable],
    real_counts: Dict[float, Dict[int, int]],
    max_attempts_per_branch: int = 50,
    debug: bool = False 
) -> Tuple[Dict[float, torch.Tensor], Tuple[Dict[int, torch.Tensor], set, Dict[int, int]]]:
    
    device = initial_points.device
    num_features = initial_points.shape[1]
    snapshots = {observation_times[0]: initial_points.clone()}
    tracked_paths, path_lineage = {}, {}
    next_path_id = 0
    current_path_ids = []
    for i in range(initial_points.shape[0]):
        path_id = next_path_id
        tracked_paths[path_id] = initial_points[i].unsqueeze(0).clone()
        current_path_ids.append(path_id)
        next_path_id += 1

    for k in range(len(observation_times) - 1):
        t_current, t_next = observation_times[k], observation_times[k+1]
        
        if debug:
            print("\n" + "="*60)
            print(f"PROBE: Entering interval [{t_current:.1f} -> {t_next:.1f}]")
            print("="*60)
        
        current_points = snapshots[t_current]
        if current_points.shape[0] == 0:
            print(f"All paths terminated at t={t_current}. Stopping generation.")
            break
      
        target_counts = real_counts.get(t_next, {}) 
        if debug:
            print(f"DEBUG [1]: Target counts for t={t_next:.1f} are: {target_counts}")
            if not target_counts:
                print("  CRITICAL DEBUG: Target count dictionary is empty! All particles will be culled by omission.")

        cluster_model = cluster_models[t_next]
        
        print(f"Propagating {len(current_path_ids)} particles from t={t_current}...")
        
        path_segments_list = sde_solver(
            timepoint_lists_for_each_stage, current_points, v_m, scale_m, delta_t, eps, k
        )
        propagated_points = path_segments_list[-1]
        
        if debug:
            if not torch.all(torch.isfinite(propagated_points)):
                print(f"  CRITICAL DEBUG [2a]: SDE produced NaN or Inf values at t={t_next:.1f}! This is a likely cause of extinction.")
            else:
                print(f"DEBUG [2a]: SDE outputs are all finite numbers. (OK)")
        
        full_segments = torch.stack([torch.stack([step[i] for step in path_segments_list]) for i in range(current_points.shape[0])])
        propagated_labels = cluster_model(propagated_points)
        
        if debug:
            unique_labels_produced = torch.unique(propagated_labels)
            print(f"DEBUG [2b]: Unique labels produced by clustering model: {unique_labels_produced.cpu().numpy()}")
            unexpected_labels = set(unique_labels_produced.cpu().numpy()) - set(target_counts.keys())
            if unexpected_labels:
                print(f"  CRITICAL DEBUG: Found unexpected labels {unexpected_labels} which are not in the target list!")

        indices_in_clusters = {j: [] for j in target_counts.keys()}
        for i, label in enumerate(propagated_labels):
            label_item = label.item()
            if label_item in indices_in_clusters:
                indices_in_clusters[label_item].append(i)
        
        if debug:
            print(f"DEBUG [3]: Particles counted per target cluster:")
            total_counted = 0
            for cid, inds in indices_in_clusters.items():
                count = len(inds)
                print(f"  - Target Cluster {cid}: Found {count} particles.")
                total_counted += count
            if total_counted == 0 and len(propagated_points) > 0:
                print("  CRITICAL DEBUG: No particles were counted for ANY target cluster. This confirms an 'Invalid Label' or 'Data Type Mismatch' issue.")

        final_points_k, final_path_ids_k = [], []
        print("Calibrating particle distribution...")
        
        for j in sorted(target_counts.keys()):
            target_count = target_counts[j]
            candidate_indices = torch.tensor(indices_in_clusters.get(j, []), dtype=torch.long)
            current_count = len(candidate_indices)
            
            print(f"  Cluster {j}: Target={target_count}, Propagated={current_count}")

            if current_count > target_count:
                print(f"    -> Culling {current_count - target_count} particles.")
                survivor_indices = candidate_indices[torch.randperm(current_count)[:target_count]]
                for idx in survivor_indices:
                    parent_path_id = current_path_ids[idx]
                    final_points_k.append(propagated_points[idx])
                    final_path_ids_k.append(parent_path_id)
                    tracked_paths[parent_path_id] = torch.cat((tracked_paths[parent_path_id], full_segments[idx]), dim=0)

            elif current_count < target_count:
                deficit = target_count - current_count
                print(f"    -> Regenerating {deficit} particles.")
                
                for idx in candidate_indices:
                    parent_path_id = current_path_ids[idx]
                    final_points_k.append(propagated_points[idx])
                    final_path_ids_k.append(parent_path_id)
                    tracked_paths[parent_path_id] = torch.cat((tracked_paths[parent_path_id], full_segments[idx]), dim=0)

                if current_count > 0:
                    parent_pool_indices = candidate_indices
                else:
                    print(f"    WARN: Cluster {j} is empty. Using all previous particles as parent pool.")
                    parent_pool_indices = torch.arange(len(current_path_ids))
                
                if debug:
                    print(f"  DEBUG [4]: For regenerating cluster {j}, parent pool size is {len(parent_pool_indices)}")
                    if len(parent_pool_indices) == 0:
                        print("    CRITICAL DEBUG: Parent pool is empty! Regeneration is impossible.")
                
                if len(parent_pool_indices) == 0:
                    print(f"    ERROR: Cannot regenerate for Cluster {j} as parent pool is empty.")
                    continue

                added_count, attempts = 0, 0
                while added_count < deficit and attempts < max_attempts_per_branch * deficit:
                    rand_idx = torch.randint(0, len(parent_pool_indices), (1,)).item()
                    parent_idx = parent_pool_indices[rand_idx]
                    parent_point = current_points[parent_idx].unsqueeze(0)
                    parent_path_id = current_path_ids[parent_idx]

                    new_segment_list = sde_solver(
                        timepoint_lists_for_each_stage, parent_point, v_m, scale_m, delta_t, eps, k,
                    )
                    
                    new_point = new_segment_list[-1]
                    
                    if cluster_model(new_point).item() == j:
                        added_count += 1
                        new_path_id = next_path_id
                        next_path_id += 1
                        
                        final_points_k.append(new_point.squeeze(0))
                        final_path_ids_k.append(new_path_id)
                        path_lineage[new_path_id] = parent_path_id
                        
                        new_segment = torch.stack([step[0] for step in new_segment_list])
                        parent_history = tracked_paths.get(parent_path_id, torch.empty(0, num_features, device=device))
                        tracked_paths[new_path_id] = torch.cat((parent_history, new_segment), dim=0)
                    
                    attempts += 1
                
                if added_count < deficit:
                    print(f"    WARN: Failed to generate all required particles for cluster {j} after {attempts} attempts. Generated {added_count}/{deficit}.")

            else: # Exactly matched
                for idx in candidate_indices:
                    parent_path_id = current_path_ids[idx]
                    final_points_k.append(propagated_points[idx])
                    final_path_ids_k.append(parent_path_id)
                    tracked_paths[parent_path_id] = torch.cat((tracked_paths[parent_path_id], full_segments[idx]), dim=0)

        if final_points_k:
            snapshots[t_next] = torch.stack(final_points_k)
            current_path_ids = final_path_ids_k
        else:
            snapshots[t_next] = torch.empty((0, num_features), device=device)
            current_path_ids = []
            
        print(f"-> End of interval. Total particles at t={t_next:.1f}: {len(current_path_ids)}")

        if debug:
            if len(current_path_ids) == 0 and len(initial_points) > 0:
                print("="*60)
                print(f"  CRITICAL DEBUG: TOTAL POPULATION EXTINCTION EVENT at t={t_next:.1f}!")
                print("="*60)

    final_surviving_path_ids = set(current_path_ids)
    
    final_lineage = {}
    for child_id, parent_id in path_lineage.items():
        if child_id in tracked_paths and parent_id in tracked_paths:
            final_lineage[child_id] = parent_id

    return snapshots, (tracked_paths, final_surviving_path_ids, final_lineage)



def calculate_target_counts_from_labels(
    observed_labels: List[torch.Tensor],
    initial_simulation_size: int,
    observation_times: List[float]
) -> Dict[float, Dict[int, int]]:

    if not observed_labels:
        raise ValueError("input `observed_labels` con not be empty")
        
    if len(observed_labels) != len(observation_times):
        raise ValueError("`observed_labels` and `observation_times` must have a same length")

    real_initial_size = len(observed_labels[0])
    if real_initial_size == 0:
        raise ValueError("the first time point must have at least one observation")


    scaling_factor = initial_simulation_size / real_initial_size

    target_counts_all_times = {}

    for i, labels_at_k in enumerate(observed_labels):
        time_k = observation_times[i]
        real_total_at_k = len(labels_at_k)

        if real_total_at_k == 0:
            target_counts_all_times[time_k] = {}
            continue

        unique_labels, real_counts_k = torch.unique(labels_at_k, return_counts=True)

        scaled_counts_k = {}
        for label, real_count in zip(unique_labels, real_counts_k):
            target_count = int(round(real_count.item() * scaling_factor))
            scaled_counts_k[label.item()] = target_count

        expected_total_n_k = int(round(real_total_at_k * scaling_factor))
        current_total_n_k = sum(scaled_counts_k.values())
        
        diff = expected_total_n_k - current_total_n_k

        if diff != 0 and scaled_counts_k:
            largest_cluster = max(scaled_counts_k, key=scaled_counts_k.get)
            scaled_counts_k[largest_cluster] += diff

        target_counts_all_times[time_k] = scaled_counts_k

    return target_counts_all_times



def generate_snapshots_at_times(
    tracked_paths: Dict[int, torch.Tensor],
    full_simulation_times: np.ndarray,
    query_times: List[float]
) -> Dict[float, torch.Tensor]:

    try:
        time_to_idx_map = {t: i for i, t in enumerate(full_simulation_times)}
    except TypeError: 
        time_to_idx_map = {t: i for i, t in enumerate(full_simulation_times)}


    new_snapshots = {}
    
   
    for t_query in query_times:

        if t_query not in time_to_idx_map:
            print(f"Warning: {t_query} not in full_simulation_times")
            continue
            
        target_idx = time_to_idx_map[t_query]

        particles_at_t_query = []
        
        for path_id, path_tensor in tracked_paths.items():

            if path_tensor.shape[0] > target_idx:
                particle_state = path_tensor[target_idx]
                particles_at_t_query.append(particle_state)
        
        if particles_at_t_query:
            new_snapshots[t_query] = torch.stack(particles_at_t_query)
        else:
            feature_dim = next(iter(tracked_paths.values())).shape[1] if tracked_paths else 2
            new_snapshots[t_query] = torch.empty((0, feature_dim))
            
    return new_snapshots