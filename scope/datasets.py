import torch
import numpy as np
from torch.utils.data import Dataset, BatchSampler
from typing import List, Dict, Tuple, Literal

# --- 1. 创建一个数据集类 ---
# 它的职责是存储所有数据，并能通过索引访问

class EvolutionDataset(Dataset):
    def __init__(self, raw_data_by_time: List[torch.Tensor], labels_by_time: List[torch.Tensor]):
        """
        将按时间组织的Tensor数据扁平化，以便通过全局索引访问。

        参数:
        raw_data_by_time: 原始数据点列表，每个元素是代表一个时间点数据的PyTorch Tensor。
        labels_by_time: 对应的聚类标签列表，每个元素也是一个PyTorch Tensor。
        """
        self.all_data_flat = []
        self.all_labels_flat = []
        self.all_times_flat = []
        self.indices_by_time = {} # 存储每个时间点对应的全局索引范围

        current_index = 0
        for t, (data, labels) in enumerate(zip(raw_data_by_time, labels_by_time)):
            if not isinstance(data, torch.Tensor) or not isinstance(labels, torch.Tensor):
                 raise TypeError(f"Inputs at index {t} must be torch.Tensors.")
            
            num_samples = data.shape[0]
            self.all_data_flat.append(data.float())
            self.all_labels_flat.append(labels.long())

            self.all_times_flat.append(torch.full((num_samples,), t, dtype=torch.long))
            
            self.indices_by_time[t] = list(range(current_index, current_index + num_samples))
            current_index += num_samples

        # 将列表拼接成一个大张量
        self.all_data_flat = torch.cat(self.all_data_flat, dim=0)
        self.all_labels_flat = torch.cat(self.all_labels_flat, dim=0)
        self.all_times_flat = torch.cat(self.all_times_flat, dim=0)

    def __len__(self):
        return len(self.all_data_flat)

    def __getitem__(self, idx):
        return {
            'data': self.all_data_flat[idx],
            # 'label': self.all_labels_flat[idx],
            'time': self.all_times_flat[idx]
        }
        

# --- 2. 创建自定义的批次采样器 ---
# 这是实现不等概率、跨时间步采样的核心

class TimeSeriesWeightedBatchSampler(BatchSampler):
    def __init__(self, dataset: EvolutionDataset, cluster_weights: Dict[Tuple[int, int], float], batch_size: int):
        """
        每次生成一个包含所有时间点样本的“大批次”索引。
        在每个时间点内部，根据聚类权重进行不均衡采样。

        参数:
        dataset: 上面定义的 EvolutionDataset 对象。
        cluster_weights: 聚类权重字典, e.g., {(t, cluster_id): weight, ...}
        batch_size: 每个时间点要采样的样本数量。
        """
        self.dataset = dataset
        self.cluster_weights = cluster_weights
        self.batch_size = batch_size
        self.num_timesteps = len(dataset.indices_by_time)
        
        # --- 预处理：为每个时间点的每个样本计算其被抽中的概率 ---
        self.weights_per_timestep = {}
        for t in range(self.num_timesteps):
            # 获取该时间点的所有全局索引和对应的聚类标签
            time_indices = self.dataset.indices_by_time[t]
            time_labels = self.dataset.all_labels_flat[time_indices]
            
            if not time_indices:
                continue

            # 创建一个权重张量，其长度等于该时间点的样本数
            sample_weights = torch.zeros(len(time_indices), dtype=torch.float)
            
            # 为每个样本赋予其所在聚类的权重
            for i, label_tensor in enumerate(time_labels):
                label = label_tensor.item()
                sample_weights[i] = self.cluster_weights.get((t, label), 0)
            
            self.weights_per_timestep[t] = sample_weights

        # 估算每个 epoch 的批次数
        # 这里简单地假设我们希望遍历所有数据一次
        total_samples = len(dataset)
        mega_batch_size = self.batch_size * self.num_timesteps
        self.num_batches = total_samples // mega_batch_size if mega_batch_size > 0 else 0

    def __iter__(self):
        for _ in range(self.num_batches):
            mega_batch = []
            # 对每个时间点进行独立采样
            for t in range(self.num_timesteps):
                if t not in self.weights_per_timestep:
                    continue

                sample_weights = self.weights_per_timestep[t]
                time_indices = self.dataset.indices_by_time[t]
                
                # 如果所有权重都为0，则退化为均匀采样
                if torch.sum(sample_weights) == 0:
                    sample_weights = torch.ones_like(sample_weights)

                # 使用 torch.multinomial 进行加权随机抽样（有放回）
                # 它返回的是相对于 time_indices 的局部索引
                local_sampled_indices = torch.multinomial(
                    sample_weights, 
                    num_samples=self.batch_size, 
                    replacement=True
                )
                
                # 将局部索引映射回全局索引
                global_sampled_indices = [time_indices[i] for i in local_sampled_indices]
                mega_batch.extend(global_sampled_indices)
            
            if mega_batch:
                yield mega_batch

    def __len__(self):
        return self.num_batches
    
    
    
from collections import defaultdict
def collate_by_timestep(batch_as_list: List[Dict[str, torch.Tensor]]) -> List[torch.Tensor]:
    """
    将一批样本按时间点分组，并打包成一个Tensor列表。
    
    参数:
    batch_as_list: DataLoader从Dataset中取出的一批样本的列表。
                   例如: [{'data': tensor, 'time': 0}, {'data': tensor, 'time': 1}, ...]

    返回:
    List[torch.Tensor]: 一个列表，第 i 个元素是 t=i 时刻所有样本数据的Tensor。
    """
    # 按时间点对数据进行分组
    data_by_time = defaultdict(list)
    for sample in batch_as_list:
        time = sample['time'].item()
        data_by_time[time].append(sample['data'])
    
    # 确定最大的时间点，以构建完整列表
    max_time = max(data_by_time.keys()) if data_by_time else -1
    
    # 将每个时间点的数据堆叠成一个Tensor
    final_batch = []
    for t in range(max_time + 1):
        if data_by_time[t]:
            # torch.stack 会将一个 tensor 列表在新的维度上堆叠起来
            final_batch.append(torch.stack(data_by_time[t], dim=0))
        else:
            # 如果某个时间点没有采到样本，可以添加一个空Tensor作为占位符
            final_batch.append(torch.tensor([]))
            
    return final_batch


"""
一个完整的、基于演化树结构进行序列采样的数据加载器框架。

该模块提供了：
1. PathAssembledDataset: 一个PyTorch数据集，其每个样本对应于演化树中的一条完整路径序列。
2. path_collate_fn: 一个配套的整理函数，用于将样本序列打包成批次。
3. 一个端到端的演示，展示了如何：
    a. 根据演化路径终点的样本量计算路径权重。
    b. 使用 WeightedRandomSampler 进行加权采样。
    c. 构建最终的 DataLoader。

此框架可以直接用于训练需要长序列输入的下游生成模型（如神经SDE、基于Transformer/RNN的动力学模型）。
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict
import random

# ==============================================================================
# 模块一：核心数据集类
# ==============================================================================

class PathAssembledDataset(Dataset):
    def __init__(self, 
                 raw_data_by_time: List[torch.Tensor], 
                 labels_by_time: List[torch.Tensor],
                 edges: List[Tuple[Any, Any, float, str]],
                 observation_times: List[float],
                 path_length: int = 3):
        """
        这个Dataset的每个"item"对应s-MSF图中的一条有效路径（演化蓝图）。
        它会根据索引返回一个由该路径上每个节点的真实样本点构成的序列。

        参数:
        raw_data_by_time: 按时间组织的原始数据Tensor列表。
        labels_by_time: 对应的聚类标签Tensor列表。
        edges: 由s-MSF生成的演化边列表。
        observation_times: 观测时间点列表。
        path_length (int): 每个训练样本序列所包含的时间节点数量。
        """
        self.observation_times = observation_times
        self.path_length = path_length
        
        # --- 预处理1：将原始数据点按 (t, cluster_id) 进行分组 ---
        # 这是一个关键的预处理步骤，可以极大地加速后续的随机采样过程。
        self.points_by_cluster = defaultdict(list)
        # 我们需要一个全局的数据张量，以便用索引直接访问，避免多次拼接。
        all_data_flat = torch.cat(raw_data_by_time, dim=0)
        
        current_offset = 0
        for t, (data, labels) in enumerate(zip(raw_data_by_time, labels_by_time)):
            for i in range(len(data)):
                cluster_id = (t, labels[i].item())
                # 注意：这里存储的是真实的Tensor数据，而不是索引，方便直接使用。
                self.points_by_cluster[cluster_id].append(all_data_flat[current_offset + i])
            current_offset += len(data)

        # --- 预处理2：从图中找出所有指定长度的“中心点路径”（演化蓝图）---
        # 构建一个从父到子的邻接表
        adj_children = defaultdict(list)
        for u, v, _, _ in edges:
            adj_children[u].append(v)
        
        self.valid_paths = []
        # 路径的起点必须是那些作为父节点出现过的节点
        start_nodes = list(adj_children.keys())
        
        # 使用深度优先搜索(DFS)来寻找所有符合长度的路径
        for start_node in start_nodes:
            # 只有当节点的起始时间允许构造出足够长的路径时才开始搜索
            if start_node[0] <= len(observation_times) - self.path_length:
                 self._find_paths_dfs(start_node, [start_node], adj_children)
        
        if not self.valid_paths:
            raise ValueError("在给定的演化图中，未能找到任何符合指定长度的路径！请检查path_length或图的连通性。")
            
        print(f"INFO: 从演化图中找到了 {len(self.valid_paths)} 条长度为 {self.path_length} 的有效路径（演化蓝图）。")

    def _find_paths_dfs(self, current_node: Tuple, current_path: List, adj: Dict):
        """递归地寻找路径"""
        if len(current_path) == self.path_length:
            self.valid_paths.append(list(current_path))
            return

        # 递归地探索子节点
        for child_node in adj.get(current_node, []):
            current_path.append(child_node)
            self._find_paths_dfs(child_node, current_path, adj)
            current_path.pop() # 回溯，以便探索其他分支

    def __len__(self) -> int:
        """数据集的大小等于“演化蓝图”的数量。"""
        return len(self.valid_paths)

    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """根据索引获取一个“演化蓝图”，并用真实数据将其“实例化”。"""
        # 1. 根据索引获取一条中心点路径
        center_path = self.valid_paths[idx]
        
        sample_sequence = []
        time_sequence = []
        
        # 2. 为路径上的每个中心点，随机抽取一个真实样本
        for node_id in center_path:
            point_candidates = self.points_by_cluster.get(node_id)
            if not point_candidates: 
                # 理论上，如果节点存在于路径中，其聚类不应为空。这是一个保护性措施。
                print(f"Warning: Cluster {node_id} has no data points. Skipping path.")
                return None 

            # 随机选择一个点
            random_point = random.choice(point_candidates)
            sample_sequence.append(random_point)
            time_sequence.append(self.observation_times[node_id[0]])
            
        return torch.stack(sample_sequence), torch.tensor(time_sequence, dtype=torch.float)

# ==============================================================================
# 模块二：批次整理函数
# ==============================================================================

def path_collate_fn(batch: List[Optional[Tuple[torch.Tensor, torch.Tensor]]]) -> Optional[Dict[str, torch.Tensor]]:
    """
    将一批 (样本序列, 时间序列) 元组打包成一个批次字典。
    """
    # 过滤掉在__getitem__中可能返回的None值
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # 将数据解包并堆叠成批次
    sequences = torch.stack([item[0] for item in batch], dim=0)
    times = torch.stack([item[1] for item in batch], dim=0)
    
    return {"sequences": sequences, "times": times}



import math
from collections import defaultdict
from typing import List, Tuple, Any, Dict, Literal
import numpy as np

# ==============================================================================
# 辅助函数 1: 计算基础路径权重
# ==============================================================================
def _calculate_holistic_path_weights(
    valid_paths: List[List[Tuple]],
    sample_sizes: Dict[Tuple[int, int], int],
    edges: List[Tuple[Any, Any, float, str]],
    log_transform_samples: bool = False
) -> List[float]:
    """
    计算每条路径的综合权重。
    权重 = 路径上所有节点贡献度之和
    贡献度 = 节点样本量 / (节点的入度(父节点数) * 节点的出度(子节点数))
    【注意】: 为清晰起见，函数名前加了下划线，表示其为内部辅助函数。
    """
    if not valid_paths:
        return []

    in_degrees = defaultdict(int)
    for _, child_node, _, _ in edges:
        in_degrees[child_node] += 1

    out_degrees = defaultdict(int)
    for parent_node, _, _, _ in edges:
        out_degrees[parent_node] += 1

    path_weights = []
    for path in valid_paths:
        current_path_weight = 0.0
        for node in path:
            node_sample_size = sample_sizes.get(node, 0)
            
            if log_transform_samples:
                contribution_base = np.log1p(node_sample_size)
            else:
                contribution_base = node_sample_size
            
            node_in_degree = in_degrees.get(node, 1)
            if node_in_degree == 0:
                node_in_degree = 1
                
            node_out_degree = out_degrees.get(node, 1)
            if node_out_degree == 0:
                node_out_degree = 1
            
            current_path_weight += contribution_base / (node_in_degree * node_out_degree)
        
        path_weights.append(current_path_weight)
        
    return path_weights

# ==============================================================================
# 辅助函数 2: 校准权重并归一化 
# ==============================================================================
def _regularize_and_normalize_weights(
    path_weights: List[float],
    weighting_strategy: Literal['sqrt_inverse', 'inverse', 'enos', 'none'] = 'sqrt_inverse',
    beta: float = 0.999,
    epsilon: float = 1e-8 
) -> List[float]:
    """
    根据不同策略对权重列表进行校正，并进行归一化，使其和为1。
    【注意】: 函数名前加下划线，并增加了 'none' 选项。
    """
    if not path_weights:
        return []

    corrected_weights = []
    for weight in path_weights:
        if weighting_strategy == 'inverse':
            correction_factor = 1.0 / (weight + epsilon)
        elif weighting_strategy == 'sqrt_inverse':
            correction_factor = 1.0 / math.sqrt(weight + epsilon)
        elif weighting_strategy == 'enos':
            correction_factor = (1.0 - beta) / (1.0 - math.pow(beta, weight) + epsilon)
        else: # 'none' or any other value
            correction_factor = 1.0
        corrected_weights.append(weight * correction_factor)
    
    total_weight = sum(corrected_weights)
    if total_weight < epsilon:
        # 如果所有权重都接近0，则平均分配
        num_weights = len(corrected_weights)
        return [1.0 / num_weights] * num_weights if num_weights > 0 else []

    return [w / total_weight for w in corrected_weights]

# ==============================================================================
# 方案二：封装后的主函数
# ==============================================================================
def calculate_endpoint_centric_weights(
    valid_paths: List[List[Tuple]],
    sample_sizes: Dict[Tuple[int, int], int],
    edges: List[Tuple[Any, Any, float, str]],
    log_transform_samples: bool = False,
    path_regularization_strategy: Literal['sqrt_inverse', 'inverse', 'enos', 'none'] = 'sqrt_inverse',
    endpoint_weighting_mode: Literal['regularized', 'sample_size', 'uniform'] = 'regularized',
    beta: float = 0.999,
    epsilon: float = 1e-8
) -> List[float]:
    """
    实现以终点为中心的二级权重分配方案。

    此函数首先确定每个终点的目标总权重，然后将该权重分配给所有通向该终点的路径。

    Args:
        valid_paths (List[List[Tuple]]): 所有合法的演化路径列表。
        sample_sizes (Dict[Tuple[int, int], int]): 每个节点(cluster_id, time_id)的样本量。
        edges (List[Tuple[Any, Any, float, str]]): 图的边列表。
        log_transform_samples (bool): 是否对节点样本量进行log变换。
        path_regularization_strategy (Literal): 在一个终点内部，用于分配不同路径权重的策略。
        endpoint_weighting_mode (Literal): 用于确定不同终点之间总权重分配的策略。
            - 'regularized': 基于终点节点的样本量，应用与路径相同的正则化策略。
            - 'sample_size': 按终点节点的原始样本量，按比例分配权重。
            - 'uniform': 所有终点平分总权重。
        beta (float): 'enos'策略的参数。
        epsilon (float): 用于防止除以零的小常数。

    Returns:
        List[float]: 一个与valid_paths等长的列表，包含每条路径最终的、全局归一化的采样权重。
    """
    if not valid_paths:
        return []

    # --- 阶段一：确定每个终点的目标权重 ---

    # 1. 将路径按终点分组，存储路径的索引
    endpoint_to_path_indices = defaultdict(list)
    for i, path in enumerate(valid_paths):
        endpoint = path[-1]
        endpoint_to_path_indices[endpoint].append(i)

    endpoints = list(endpoint_to_path_indices.keys())
    
    # 2. 根据所选策略，计算每个终点的基础“得分”
    base_endpoint_scores = []
    if endpoint_weighting_mode == 'uniform':
        base_endpoint_scores = [1.0] * len(endpoints)
    elif endpoint_weighting_mode == 'sample_size':
        base_endpoint_scores = [float(sample_sizes.get(ep, 0)) for ep in endpoints]
    elif endpoint_weighting_mode == 'regularized':
        # 'regularized'模式也基于样本量，但后续会经过regularize步骤
        base_endpoint_scores = [float(sample_sizes.get(ep, 0)) for ep in endpoints]
    
    # 3. 对终点得分进行正则化和归一化，得到最终的终点权重分布
    #    注意：对于'uniform'和'sample_size'模式，我们使用'none'策略，即只做归一化。
    #    对于'regularized'模式，我们使用与路径相同的正则化策略。
    endpoint_reg_strategy = 'none'
    if endpoint_weighting_mode == 'regularized':
        endpoint_reg_strategy = path_regularization_strategy

    target_endpoint_distribution = _regularize_and_normalize_weights(
        base_endpoint_scores,
        weighting_strategy=endpoint_reg_strategy,
        beta=beta,
        epsilon=epsilon
    )
    
    target_endpoint_weights = dict(zip(endpoints, target_endpoint_distribution))

    # --- 阶段二：将终点权重分配给其下的各个路径 ---

    final_path_weights = [0.0] * len(valid_paths)

    # 遍历每个终点及其对应的路径
    for endpoint, path_indices in endpoint_to_path_indices.items():
        # 获取该终点的目标总权重
        target_weight = target_endpoint_weights[endpoint]
        
        # 如果该终点的目标权重为0，则其下所有路径权重也为0，跳过计算
        if target_weight < epsilon:
            continue

        # 提取当前终点下的所有路径
        sub_paths = [valid_paths[i] for i in path_indices]
        
        # a. 对这个子集内的路径，计算其“基础”权重
        local_holistic_weights = _calculate_holistic_path_weights(
            sub_paths, sample_sizes, edges, log_transform_samples
        )

        # b. 对这些基础权重进行正则化和归一化，得到一个局部分布
        local_path_distribution = _regularize_and_normalize_weights(
            local_holistic_weights,
            weighting_strategy=path_regularization_strategy,
            beta=beta,
            epsilon=epsilon
        )

        # c. 将终点的目标总权重，按局部分布分配给每条路径
        for i, local_weight in enumerate(local_path_distribution):
            original_path_index = path_indices[i]
            final_path_weights[original_path_index] = target_weight * local_weight
            
    return final_path_weights
