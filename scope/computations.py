import torch
import numpy as np
from typing import List, Dict, Tuple, Union



def Interp_t(timepoint_lists_for_each_stage,eps,x_0,x_1,stage_index,bondary_constrain=False):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    B=x_0.shape[0]
    d=x_0.shape[1]
    assert B%2 == 0
    if bondary_constrain:
        t1=torch.from_numpy(np.random.choice(timepoint_lists_for_each_stage[stage_index][:-1],int(B/2))).reshape(-1,1).to(device)
        t2=torch.from_numpy(np.random.choice(timepoint_lists_for_each_stage[stage_index][1:],int(B/2))).reshape(-1,1).to(device)
        t=torch.cat([t1,t2],dim=0)
    else:
        t=torch.from_numpy(np.random.choice(timepoint_lists_for_each_stage[stage_index],B)).reshape(-1,1).to(device)
    Z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=1, size=d) for i in range(B)])).to(device)
    t_ceil = timepoint_lists_for_each_stage[stage_index][-1]
    t_floor = timepoint_lists_for_each_stage[stage_index][0]
    if callable(eps):
        eps_to_use = t.cpu().apply_(eps).to(device)
    else:
        eps_to_use = eps
    x_t=((t_ceil-t)/(t_ceil-t_floor))*x_0+((t-t_floor)/(t_ceil-t_floor))*x_1+torch.sqrt(eps_to_use*(t-t_floor)*((t_ceil-t)/(t_ceil-t_floor)))*Z
    return x_t,t,t_ceil,t_floor




def generate_one_stage(timepoint_lists_for_each_stage,x_start,v_m,scale_m,delta_t,eps,stage_index,forward=True):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    v_m.to(device)
    scale_m.to(device)
    B=x_start.shape[0]
    d=x_start.shape[1]
    path=[]
    path.append(x_start.detach().cpu())
    time_list=timepoint_lists_for_each_stage[stage_index]
    if not forward:
        time_list = time_list[::-1]
    t_floors_stage=timepoint_lists_for_each_stage[stage_index][0]
    t_floors_stage=torch.from_numpy(np.repeat(t_floors_stage,B)).reshape(-1,1).to(device)

    for t_new in time_list[:-1]:
        t=torch.from_numpy(np.repeat(t_new,B)).reshape(-1,1).double().to(device)
        x_t=path[-1].to(device)
        # if t_new != time_list[-2]:
        z=torch.from_numpy(np.array([np.random.normal(loc=0, scale=delta_t, size=d) for i in range(B)])).to(device)
        if callable(eps):
            eps_to_use = t.cpu().apply_(eps).to(device)
        else:
            # print(eps)
            eps_to_use = torch.from_numpy(np.repeat(eps,B)).reshape(-1,1).double().to(device)
        del_x_t=v_m(x_t,scale_m(t,t_floors_stage))*delta_t+torch.sqrt(eps_to_use)*z
        # else:
        #     del_x_t=v_m(x_t,scale_m(t,t_floors_stage))*delta_t
        path.append((x_t+del_x_t).detach().cpu())
    return path





def generate_path(timepoint_lists_for_each_stage,x_start,v_m,scale_m,delta_t,eps,forward=True):
    path = []
    stage_index_list = list(range(len(timepoint_lists_for_each_stage)))
    path.append(x_start.detach().cpu())
    if not forward:
        stage_index_list = stage_index_list[::-1]
    for stage_id in stage_index_list:
        path += generate_one_stage(timepoint_lists_for_each_stage,path[-1],v_m,scale_m,delta_t,eps,stage_id,forward=True)[1:]
    return path



def apply_ema_to_model(model,ema_params,decay):
    for name, param in model.named_parameters():
        ema_params[name].mul_(decay).add_(param, alpha=1 - decay)
        param.data.copy_(ema_params[name])



def max_adjacent_covariance_diagonal_differences(tensor_list):
    result = []
    num_tensors = len(tensor_list)
    for i in range(num_tensors - 1):
        cov_i = torch.cov(tensor_list[i].T)
        cov_j = torch.cov(tensor_list[i + 1].T)
        diag_i = torch.diag(cov_i)
        diag_j = torch.diag(cov_j)
        diff = torch.abs(diag_i - diag_j)
        result.append(torch.max(diff).item())
    return result



def eps_scalar(eps,snr=0.75, min_eps=0.1):
    return (1-snr) * eps + min_eps 



def piecewise_eps_function(time_points, values, adjust=eps_scalar, snr=0.8, min_eps=0.1):
    def func(x):
        v = None
        for i in range(len(values)):
            if time_points[i] <= x < time_points[i + 1]:
                v = values[i]/(time_points[i + 1]-time_points[i])
            elif x == time_points[-1]:
                v = values[-1]/(time_points[-1]-time_points[-2])

        if adjust is not None and v is not None:
            return adjust(v,snr = snr, min_eps=min_eps)
        else:
            return v
    return func



def count_cluster_samples(all_labels_by_time: List[torch.Tensor]) -> Dict[Tuple[int, int], int]:

    sample_sizes = {}

    for t, labels_at_t in enumerate(all_labels_by_time):
        
        if not isinstance(labels_at_t, torch.Tensor):
            raise TypeError(f"Input at index {t} must be a torch.Tensor, but got {type(labels_at_t)}")
        
        unique_labels, counts = torch.unique(labels_at_t, return_counts=True)
        

        for label, count in zip(unique_labels, counts):
            cluster_id = label.item()
            sample_count = count.item()
            
            sample_sizes[(t, cluster_id)] = sample_count
            
    return sample_sizes


