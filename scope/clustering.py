import torch
import numpy as np
from typing import List, Dict, Tuple, Union




def kmeans(X, k, device, n_init=10, tol=1e-4, max_iter=300):
    if k == 0:  
        raise ValueError("k must be at least 1")
    
    best_sse = float('inf')
    best_centroids = best_labels = None
    N, D = X.shape
    
    for _ in range(n_init):
        centroids = kmeans_plus_plus_init(X, k, device)
        prev_centroids = torch.zeros_like(centroids, device=device)
        
        for iter_idx in range(max_iter):
            assert X.device == device, "Input tensor must be on the same device as centroids"
            
            dists = torch.cdist(X, centroids)  # (N, k)
            labels = dists.argmin(dim=1)       # (N,)
            
            torch._assert(
                labels.max() < k,
                "Labels exceed cluster count"
            )
            
            cluster_sizes = torch.bincount(labels, minlength=k)
            empty_clusters = torch.where(cluster_sizes == 0)[0]
            
            if empty_clusters.numel() > 0:
                for cluster_id in empty_clusters:
                    cluster_dist = dists[:, cluster_id]
                    farthest_idx = cluster_dist.argmax()  
                    centroids[cluster_id] = X[farthest_idx]  
                
                dists = torch.cdist(X, centroids)
                labels = dists.argmin(dim=1)
                cluster_sizes = torch.bincount(labels, minlength=k)
                
                remaining_empty = torch.where(cluster_sizes == 0)[0]
                if remaining_empty.numel() > 0:
                    new_centers = kmeans_plus_plus_init(X, remaining_empty.numel(), device)
                    centroids[remaining_empty] = new_centers.to(device)  
            
            mask = cluster_sizes > 0  # (k,)
            new_centroids = torch.zeros_like(centroids, device=device)
            new_centroids[mask] = torch.stack([
                X[labels == i].mean(dim=0) for i in range(k) if mask[i]
            ])
            new_centroids[~mask] = centroids[~mask]  
            
            if torch.norm(new_centroids - centroids) < tol:
                break
            centroids = new_centroids

        final_dists = torch.cdist(X, centroids)
        final_labels = final_dists.argmin(dim=1)
        
        final_labels = torch.clamp(final_labels, 0, k-1) 
        
        torch._assert(
            (final_labels < k).all(),
            "Final labels contain invalid cluster IDs"
        )

        current_sse = final_dists[torch.arange(N, device=device), final_labels].sum().item()
        if current_sse < best_sse:
            best_sse = current_sse
            best_centroids = centroids
            best_labels = final_labels

    return best_centroids, best_labels





from sklearn.metrics import silhouette_score, calinski_harabasz_score

def kmeans_auto(tensors, max_k: Union[List, int], n_init: int = 5, tol: float = 1e-4, least_Ks=None, Ks=None, method='elbow') -> tuple:
    
    if not tensors:
        return [], [], []

    # 维度校验
    base_dim = tensors[0].shape[1]
    for t in tensors:
        if t.shape[1] != base_dim:
            raise ValueError("All Tensor mast have the same feature dimension D")
    if least_Ks is None:
        least_Ks = [1] * len(tensors)
    if type(least_Ks) is int:
        least_Ks = [least_Ks] * len(tensors)
    if method != 'elbow':
        least_Ks = [max(k,2) for k in least_Ks]
    cluster_centers = []
    labels_list = []
    best_k_list = []

    for ind, X in enumerate(tensors):
        N, D = X.shape
        device = X.device

        if N <= 1:  
            if N == 0:
                cluster_centers.append(torch.zeros(0, D, device=device))
                labels_list.append(torch.zeros(0, dtype=torch.int64, device=device))
                best_k_list.append(0)
                continue
            cluster_centers.append(X.unsqueeze(0))
            labels_list.append(torch.zeros(1, dtype=torch.int64, device=device))
            best_k_list.append(1)
            continue

        if Ks is None:
            if type(max_k) is not int:
                max_possible_k = min(max_k[ind], N - 1)
            else:
                max_possible_k = min(max_k, N - 1)
            if max_possible_k < 1:
                cluster_centers.append(X.unsqueeze(0))
                labels_list.append(torch.zeros(N, dtype=torch.int64, device=device))
                best_k_list.append(1)
                continue

            if method == 'elbow':
                sse = torch.zeros(max_possible_k, device=device)
                for k in range(least_Ks[ind], max_possible_k + 1):
                    centroids, labels = kmeans(X, k, device, n_init, tol)
                    dists = torch.cdist(X, centroids)
                    sse[k - 1] = dists[torch.arange(N), labels].sum()

                delta = sse[1:] - sse[:-1]
                if len(delta) < 1:
                    best_k = 1
                else:
                    second_diff = delta[1:] - delta[:-1]
                    elbow_idx = (second_diff < 0).nonzero(as_tuple=True)[0]
                    best_k = (elbow_idx[0] + 2).item() if elbow_idx.numel() else 1

                best_k = min(best_k, max_possible_k)
                best_k = max(best_k, 1)  

            elif method == 'silhouette':
                best_score = -1
                best_k = 1
                for k in range(least_Ks[ind], max_possible_k + 1):
                    centroids, labels = kmeans(X, k, device, n_init, tol)
                    score = silhouette_score(X.cpu().numpy(), labels.cpu().numpy())
                    if score > best_score:
                        best_score = score
                        best_k = k

            elif method == 'gap_statistic':
                def generate_reference_data(X, n_samples):
                    mins = X.min(dim=0)[0].cpu().numpy()
                    maxs = X.max(dim=0)[0].cpu().numpy()
                    return torch.tensor(np.random.uniform(mins, maxs, size=(n_samples, X.shape[1])), device=X.device)

                n_refs = 5
                gaps = []
                for k in range(least_Ks[ind], max_possible_k + 1):
                    centroids, labels = kmeans(X, k, device, n_init, tol)
                    dists = torch.cdist(X, centroids)
                    W_k = dists[torch.arange(N), labels].sum().cpu().item()

                    ref_W_k = []
                    for _ in range(n_refs):
                        ref_X = generate_reference_data(X, N)
                        ref_centroids, ref_labels = kmeans(ref_X, k, device, n_init, tol)
                        ref_dists = torch.cdist(ref_X, ref_centroids)
                        ref_W_k.append(ref_dists[torch.arange(N), ref_labels].sum().cpu().item())

                    ref_W_k_mean = np.mean(np.log(ref_W_k))
                    gap = ref_W_k_mean - np.log(W_k+1e-5)
                    gaps.append(gap)

                best_k = np.argmax(gaps) + least_Ks[ind]

            elif method == 'calinski_harabasz':
                best_score = -1
                best_k = 1
                for k in range(least_Ks[ind], max_possible_k + 1):
                    centroids, labels = kmeans(X, k, device, n_init, tol)
                    score = calinski_harabasz_score(X.cpu().numpy(), labels.cpu().numpy())
                    if score > best_score:
                        best_score = score
                        best_k = k

            else:
                raise ValueError(f"Unsupported method: {method}，please choose from 'elbow', 'silhouette', 'gap_statistic', 'calinski_harabasz'")

        else:
            best_k = Ks[ind]

        centroids, labels = kmeans(X, best_k, device, n_init, tol)

        cluster_centers.append(centroids)
        labels_list.append(labels)
        best_k_list.append(best_k)

    return cluster_centers, labels_list, best_k_list



def kmeans_plus_plus_init(X: torch.Tensor, k: int, device) -> torch.Tensor:

    N, D = X.shape
    centroids = torch.zeros(k, D, device=device).double()
    centroids[0] = X[torch.randint(0, N, (1,), device=device)]
    
    for i in range(1, k):
        dists = torch.cdist(X, centroids[:i])  # [N, i]
        min_dists = dists.min(dim=1).values + 1e-5  # [N]
        probs = (min_dists ** 2) / (min_dists ** 2).sum()
        idx = torch.multinomial(probs, 1).item()
        centroids[i] = X[idx]
    
    return centroids



def assign_labels(samples, centers):

    samples_expanded = samples.unsqueeze(1)
    centers_expanded = centers.unsqueeze(0)

    # distances_sq.shape: (n_samples, n_clusters)
    distances_sq = torch.sum((samples_expanded - centers_expanded.to(samples_expanded.device)) ** 2, dim=2)

    # labels.shape: (n_samples,)
    labels = torch.argmin(distances_sq, dim=1)

    return labels




def calculate_label_ratios(labels_list: list[torch.Tensor]) -> list[torch.Tensor]:

    result = []
    for labels in labels_list:
        if labels.numel() == 0:
            result.append(torch.tensor([]))
            continue
            
        max_label = int(labels.max().item())
        bin_count = torch.bincount(labels, minlength=max_label + 1)
        ratios = bin_count.float() / labels.numel()
        result.append(torch.round(ratios * 10000) / 10000)  
        
    return result


