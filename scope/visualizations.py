import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Dict, Tuple, Any, Optional, Union



def visualize_full_evolution(
    raw_data: List[np.ndarray],
    centers: List[np.ndarray],
    all_points_map: Dict[Tuple[int, int], np.ndarray],
    edges: List[Tuple[Any, Any, float, str]],
    time_labels: Optional[List[str]] = None,
    umap_model: Optional[Any] = None,
    legend_fontsize: int = 10,
    save_path: Optional[str] = None,
    adjust_upside_down: Optional[bool] = False
):
    
    num_timesteps = len(centers)
    fig, ax = plt.subplots(figsize=(12, 12))
    colors = cm.viridis(np.linspace(0, 1, num_timesteps))

    if time_labels is None:
        time_labels = ['Time ' + str(t) for t in range(num_timesteps)]

    print("INFO: Preparing data for visualization...")
    if umap_model is not None:
        print("INFO: UMAP model provided. Transforming high-dimensional data to 2D...")
        all_raw_points_list = [d for d in raw_data if d.size > 0]
        temp_center_list = [c for c in centers if c.size > 0]
        if not all_raw_points_list and not temp_center_list:
            print("Warning: No data to visualize.")
            return
        raw_lengths = [len(d) for d in all_raw_points_list]
        center_lengths = [len(c) for c in temp_center_list]
        points_to_transform = np.vstack(all_raw_points_list + temp_center_list)
        transformed_points = umap_model.transform(points_to_transform)
        if adjust_upside_down:
            transformed_points[:, 1] = -transformed_points[:, 1] + 9 
        raw_data_2d, centers_2d, all_points_map_2d = [], [], {}
        current_idx = 0
        for length in raw_lengths:
            raw_data_2d.append(transformed_points[current_idx : current_idx + length])
            current_idx += length
        center_idx = 0
        for t in range(len(centers)):
            if centers[t].size > 0:
                length = center_lengths[center_idx]
                centers_2d.append(transformed_points[current_idx : current_idx + length])
                for j in range(length):
                    if (t, j) in all_points_map:
                         all_points_map_2d[(t, j)] = transformed_points[current_idx + j]
                current_idx += length; center_idx += 1
            else: centers_2d.append(np.array([]))
    else:

        print("INFO: No UMAP model provided. Plotting the first two dimensions.")
        raw_data_2d = [d[:, :2] if d.ndim > 1 and d.shape[1] > 1 else np.array([]) for d in raw_data]
        centers_2d = [c[:, :2] if c.ndim > 1 and c.shape[1] > 1 else np.array([]) for c in centers]
        all_points_map_2d = {k: v[:2] for k, v in all_points_map.items()}

    for t in range(num_timesteps):
        if t < len(raw_data_2d) and raw_data_2d[t].size > 0:
            ax.scatter(raw_data_2d[t][:, 0], raw_data_2d[t][:, 1], color=colors[t], s=10, alpha=0.2, zorder=1)
    edge_styles = {'backbone': {'color': 'black', 'linestyle': '-'},'rescue_child': {'color': '#0077b6', 'linestyle': '--'},'rescue_parent': {'color': '#d00000', 'linestyle': ':'}}
    for u, v, weight, edge_type in edges:
        pos_u, pos_v = all_points_map_2d.get(u), all_points_map_2d.get(v)
        if pos_u is not None and pos_v is not None:
            style = edge_styles.get(edge_type, {'color': 'gray', 'linestyle': ':'})
            ax.plot([pos_u[0], pos_v[0]], [pos_u[1], pos_v[1]], **style, linewidth=1.5, alpha=0.8, zorder=2)
    for t in range(num_timesteps):
        if t < len(centers_2d) and centers_2d[t].size > 0:
            ax.scatter(centers_2d[t][:, 0], centers_2d[t][:, 1], color=colors[t], s=100, ec='white', linewidth=1.5, zorder=3, label=f'{time_labels[t]}')

    ax.set_title('Evolutionary Graph of Cell States', fontsize=16)
    ax.set_xlabel('Dimension 1 (UMAP or Original)', fontsize=12)
    ax.set_ylabel('Dimension 2 (UMAP or Original)', fontsize=12)


    legend_main = ax.legend(
        title='Cluster Centers', 
        bbox_to_anchor=(1.02, 1), 
        loc='upper left',
        fontsize=legend_fontsize  
    )
    ax.add_artist(legend_main)
    
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Backbone'),
        Line2D([0], [0], color='#0077b6', linestyle='--', lw=2, label='Child Rescue'),
        Line2D([0], [0], color='#d00000', linestyle=':', lw=2, label='Parent Rescue')
    ]
    ax.legend(
        handles=legend_elements, 
        title='Edge Types', 
        bbox_to_anchor=(1.02, 0), 
        loc='lower left',
        fontsize=legend_fontsize 
    )

    ax.grid(True, linestyle='--', alpha=0.2)
    if save_path is not None:
        plt.savefig(
            save_path+'.svg', 
            format='svg',
            dpi=300,                
            bbox_inches='tight',    
            pad_inches=0.1          
        )
    plt.show()



def visualize_populations(
    raw_data: List[torch.Tensor],
    observed_time_points: List[float],
    cluster_centers: Optional[List[torch.Tensor]] = None,
    umap_model: Optional[Any] = None,
    adjust_equal:Optional[bool] = False,
    save_path: Optional[str] = None,
    adjust_upside_down: Optional[bool] = False
    ):

    for t in range(len(raw_data)):
        x_seq=torch.cat(raw_data).cpu().numpy()
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.tab20
        xu = umap_model.transform(x_seq) if umap_model is not None else x_seq
        xu_ = umap_model.transform(raw_data[t].cpu().numpy()) if umap_model is not None else raw_data[t].cpu().numpy()
        ax.scatter(xu[:,0], xu[:,1] if not adjust_upside_down else -xu[:,1] + 9, s=1, color = 'gray')
        ax.scatter(x=xu_[:, 0], y=xu_[:, 1] if not adjust_upside_down else -xu_[:,1] + 9, s=1, label='real ' + str(observed_time_points[t]), color='#D35400', alpha=0.5)
        if cluster_centers is not None:
            xv = umap_model.transform(cluster_centers[t].cpu().numpy()) if umap_model is not None else cluster_centers[t].cpu().numpy()
            x = xv[:, 0]
            y = xv[:, 1]
            labels = [str(i) for i in range(len(cluster_centers[t]))]
            ax.scatter(x=x, y=y if not adjust_upside_down else -y + 9, s=10, alpha=1,  marker='x', color='blue')
            for i, label in enumerate(labels):
                plt.annotate(label, (x[i], y[i] if not adjust_upside_down else -y[i] + 9), textcoords="offset points", xytext=(0,10), ha='center')
        ax.legend(fontsize=15, borderpad=1, loc='center left', bbox_to_anchor=(1, 0.5), markerscale=5)
        ax.legend(markerscale=3,loc="lower left")
        if adjust_equal:
            ax.set_aspect('equal', adjustable='box')
        if save_path is not None:
            plt.savefig(
                save_path+str(observed_time_points[t])+'.svg', 
                format='svg',
                dpi=300,                
                bbox_inches='tight',    
                pad_inches=0.1          
            )
        plt.show()


def visualize_generated_populations(
    raw_data: List[torch.Tensor],
    generated_data: List[torch.Tensor],
    observed_time_points: List[float],
    umap_model: Optional[Any] = None,
    adjust_equal:Optional[bool] = False,
    save_path: Optional[str] = None,
    adjust_upside_down: Optional[bool] = False
    ):
    x_seq=torch.cat(raw_data).cpu().numpy()
    color_lst = ['#377EB8','#D35400','#8E44AD']
    label_lst= ['WM','Layer 6','Layer 5','Layer 4','Layer 3','Layer 2','Layer 1']
    step_each = (len(generated_data)-1)//(observed_time_points[-1] - observed_time_points[0])
    for t in range(1,len(raw_data)):
        fig, ax = plt.subplots(figsize=(6, 6))
        cmap = plt.cm.tab20
        xu = umap_model.transform(x_seq) if umap_model is not None else x_seq
        xu_generated = umap_model.transform(generated_data[int(step_each*(observed_time_points[t] - observed_time_points[0]))].cpu().numpy()) if umap_model is not None else generated_data[int(step_each*(observed_time_points[t] - observed_time_points[0]))].cpu().numpy()
        ax.scatter(xu[:,0], xu[:,1] if not adjust_upside_down else -xu[:,1] + 9, s=1, color = 'gray')
        if len(raw_data) <=3:
            for t_ in range(t+1):    
                xu_ = umap_model.transform(raw_data[t_].cpu().numpy()) if umap_model is not None else raw_data[t_].cpu().numpy()
                ax.scatter(x=xu_[:, 0], y=xu_[:, 1] if not adjust_upside_down else -xu_[:,1] + 9, s=1, label='real ' + str(observed_time_points[t_]), color=color_lst[t_], alpha=0.5)
        else:
            xu_ = umap_model.transform(raw_data[t].cpu().numpy()) if umap_model is not None else raw_data[t].cpu().numpy()
            #ax.scatter(x=xu_[:, 0], y=xu_[:, 1] if not adjust_upside_down else -xu[:,1] + 9, s=1, label='real ' + str(observed_time_points[t]), color='#D35400', alpha=0.5)
            ax.scatter(x=xu_[:, 0], y=xu_[:, 1] if not adjust_upside_down else -xu[:,1] + 9, s=1, label='real ' + label_lst[t], color='#D35400', alpha=0.5)
        #ax.scatter(xu_generated[:,0], xu_generated[:,1] if not adjust_upside_down else -xu_generated[:,1] + 9, s = 1, color = '#FFBF00',label='fake ' + str(observed_time_points[t]))
        ax.scatter(xu_generated[:,0], xu_generated[:,1] if not adjust_upside_down else -xu_generated[:,1] + 9, s = 1, color = '#FFBF00',label='fake ' + label_lst[t])
        ax.legend(markerscale=3,loc="lower left")
        if adjust_equal:
            ax.set_aspect('equal', adjustable='box')
        if save_path is not None:
            plt.savefig(
                save_path+str(observed_time_points[t])+'.pdf', 
                format='pdf',
                dpi=300,                
                bbox_inches='tight',    
                pad_inches=0.1          
            )
        plt.show()
            
            
def visualize_drift_func(
    raw_data: List[torch.Tensor],
    raw_dataframe: Union[pd.DataFrame, np.ndarray],
    sb_object: Any,
    time_labels: np.ndarray,
    umap_model: Optional[Any] = None,
    xg: Optional[float] = 25,
    yg: Optional[float] = 25,
    xg_lower_bound: Optional[float] = -5,
    xg_upper_bound: Optional[float] = 15,
    yg_lower_bound: Optional[float] = -5,
    yg_upper_bound: Optional[float] = 15,
    scale: Optional[float] = 2,
    width: Optional[float] = .004,
    adjust_equal:Optional[bool] = False,
    fine_tuned:Optional[bool] = False,
    save_path: Optional[str] = None,
    adjust_upside_down: Optional[bool] = False
    ):
    device = raw_data[0].device
    fig, ax = plt.subplots(figsize = (12,12))
    x_seq=torch.cat(raw_data).cpu().numpy()
    xu = umap_model.transform(x_seq) if umap_model is not None else x_seq
    ax.scatter(xu[:,0], xu[:,1] if not adjust_upside_down else -xu[:,1] + 9, s=1, color = 'gray')

    xgrid = np.linspace(xg_lower_bound, xg_upper_bound, xg)
    ygrid = np.linspace(yg_lower_bound, yg_upper_bound, yg)

    xus = []
    xvs = []
    for xi in range(xg-1):
        for yi in range(yg-1):

            xmin, xmax = xgrid[xi], xgrid[xi+1]
            ymin, ymax = ygrid[yi], ygrid[yi+1]

            in_x = (xu[:,0] > xmin) & (xu[:,0] < xmax)
            in_y = (xu[:,1] > ymin) & (xu[:,1] < ymax)
            in_box = in_x & in_y

            if in_box.sum() > 3:
                ix = np.random.choice(np.where(in_box)[0], max(3, int(in_box.sum() * 0.002)))
                if isinstance(raw_dataframe, pd.DataFrame): 
                    xv = torch.from_numpy(raw_dataframe.values[ix,:]).double().to(device)
                elif isinstance(raw_dataframe, np.ndarray):
                    xv = torch.from_numpy(raw_dataframe[ix,:]).double().to(device)
                else:
                    print("Invalid data type")
                sb_object.scale_m_fore.to(device)
                if fine_tuned:
                    sb_object.v_fore_fine_tuned.to(device)
                    xv_g = xv + sb_object.v_fore_fine_tuned(xv,
                                                    sb_object.scale_m_fore(torch.from_numpy(time_labels[ix]).reshape(-1,1).to(device),torch.from_numpy(time_labels[ix]).reshape(-1,1).to(device)))
                else:
                    sb_object.v_fore.to(device)
                    xv_g = xv + sb_object.v_fore(xv,
                                                    sb_object.scale_m_fore(torch.from_numpy(time_labels[ix]).reshape(-1,1).to(device),torch.from_numpy(time_labels[ix]).reshape(-1,1).to(device)))
                xvs.append(xv_g.detach().cpu().numpy())
                if umap_model is not None:
                    xu_ = umap_model.transform(xv.detach().cpu().numpy())  
                else:
                    xu_ = xv.detach().cpu().numpy()[:,:2]
                if adjust_upside_down:
                    xu_[:,1] = - xu_[:,1] + 9
                xus.append(xu_)
    xvs = np.concatenate(xvs)
    xus = np.concatenate(xus)
    xv_ = umap_model.transform(xvs) if umap_model is not None else xvs
    if adjust_upside_down:
        xv_[:,1] = - xv_[:,1] + 9
    xvs = xv_ - xus 
    xvs = xvs / np.linalg.norm(xvs, axis = 1)[:,np.newaxis]
    ax.quiver(xus[:,0], xus[:,1], xvs[:,0], xvs[:,1], scale = scale, scale_units = 'xy', width = width)
    if adjust_equal:
        ax.set_aspect('equal', adjustable='box')
    if save_path is not None:
        plt.savefig(
            save_path+'.pdf', 
            format='pdf',
            dpi=300,                
            bbox_inches='tight',    
            pad_inches=0.1          
        )
    plt.show()
            
            
            
def visualize_generated_trajectories(
    raw_data: List[torch.Tensor],
    generated_data: List[torch.Tensor],
    n_trajectories: Optional[int] = 30,
    umap_model: Optional[Any] = None,
    adjust_equal:Optional[bool] = False,
    save_path: Optional[str] = None,
    adjust_upside_down: Optional[bool] = False
):
    steps = len(generated_data) - 1
    sim=torch.stack(generated_data)
    x_seq=torch.cat(raw_data).cpu().numpy()
    xu = umap_model.transform(x_seq) if umap_model is not None else x_seq
    fig, ax = plt.subplots(figsize = (12, 10))
    ax.scatter(xu[:,0], xu[:,1] if not adjust_upside_down else -xu[:,1] + 9, s=1, color = 'gray')
    c = np.arange(steps+1)
    for i in np.random.randint(0, sim.shape[1], size = n_trajectories):
        xu_ = umap_model.transform(sim[:,i,:]) if umap_model is not None else sim[:,i,:]
        sax = ax.scatter(xu_[:,0], xu_[:,1] if not adjust_upside_down else -xu_[:,1] + 9, c = c)
        ax.plot(xu_[:,0], xu_[:,1] if not adjust_upside_down else -xu_[:,1] + 9, '-', color = 'k', linewidth = 0.2)
    plt.colorbar(sax, shrink = 0.9)
    ax.set_xlabel('UMAP1')
    ax.set_ylabel('UMAP2')
    if adjust_equal:
        ax.set_aspect('equal', adjustable='box')
    if save_path is not None:
        plt.savefig(
            save_path+'.pdf', 
            format='pdf',
            dpi=300,                
            bbox_inches='tight',    
            pad_inches=0.1          
        )
    plt.show()