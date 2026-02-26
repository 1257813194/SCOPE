import torch
import numpy as np
import copy
import scope.networks
import os
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler
from collections import defaultdict

import scope.computations
import scope.backbone_train
import scope.fine_tune
import scope.datasets as ds
import scope.generation_modify as md


class sb_muti_model(object):
    def __init__(self,pi_list,timepoints,N_pretraining=1000,N_finetuning=1000,B=128,steps=60,eps=10,backbone_lr=1e-3,finetuning_lr=1e-5,
    early_stop=False,patience=10,ema=False,decay=0.8,lambda_=1e-3,save=False,plot_loss=True,save_path='model_history',record_gap=10,t_size=32,hiden_size=64,n_layers=4,prematched=False,label_list=None, edges=None, weighting_strategy='sqrt_inverse', beta=0.999):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.length=len(pi_list)
        self.pi=[TensorDataset(pi_list[i]) for i in range(self.length)]
        self.timepoints = timepoints
        self.N_pretraining=N_pretraining
        self.N_finetuning=N_finetuning
        self.B=B
        self.b=int(B/2)
        self.d=pi_list[0].shape[1]
        self.steps=steps
        self.eps=eps
        self.backbone_lr=backbone_lr
        self.finetuning_lr=finetuning_lr
        self.limit=1e-8
        
        self.epoch=1600
        self.delta_t = (timepoints[-1] - timepoints[0])/self.steps
        
        self.early_stop = early_stop
        self.patience = patience
        self.ema = ema
        self.decay=decay
        
        self.lambda_ = lambda_
        self.t_list=list(np.arange(timepoints[0],timepoints[-1],self.delta_t)) + [timepoints[-1]]
        self.save = save
        self.plot_loss = plot_loss
        self.save_path = save_path
        self.record_gap = record_gap
        
        self.prematched = prematched
        self.edges = edges
        if self.prematched:
            self.prematched_dataset = ds.PathAssembledDataset(pi_list,label_list,self.edges,timepoints,path_length=self.length)
            path_weights = []
            sample_sizes = scope.computations.count_cluster_samples(label_list)
            path_weights = ds.calculate_endpoint_centric_weights(
                self.prematched_dataset.valid_paths, 
                sample_sizes, 
                edges,
                log_transform_samples=False,
                path_regularization_strategy=weighting_strategy, 
                beta=beta,
                endpoint_weighting_mode='regularized'  
            )
            special_weight = defaultdict(float)
            for path, weight in zip(self.prematched_dataset.valid_paths, path_weights):
                special_weight[path[-1][-1]] += weight
            print(special_weight)
                
                        
            VIRTUAL_EPOCH_SIZE = pi_list[-1].shape[0] 
            VIRTUAL_EPOCH_SIZE = VIRTUAL_EPOCH_SIZE if VIRTUAL_EPOCH_SIZE % 2 == 0 else VIRTUAL_EPOCH_SIZE + 1
            sampler = WeightedRandomSampler(
                weights=torch.DoubleTensor(path_weights),
                num_samples=VIRTUAL_EPOCH_SIZE,
                replacement=True  
            )
            BATCH_SIZE = self.B
            self.path_dataloader = DataLoader(
                self.prematched_dataset,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                collate_fn=ds.path_collate_fn
            )
                

        os.makedirs(self.save_path, exist_ok=True)


        self.t_lists_stage=[[] for i in range(len(self.timepoints)-1)]
        for t in self.t_list:
            for t_index,t_point in enumerate(self.timepoints[1:]):
                if t >= t_point:
                    continue
                else:
                    self.t_lists_stage[t_index].append(t)
                    break
        for stage,t_lst in enumerate(self.t_lists_stage):
            t_lst.append(float(self.timepoints[stage+1]))

        self.v_fore=scope.networks.UNetWithLinear(x_size=self.d,t_size=t_size,output_size=self.d,hidden_size=hiden_size,n_layers=n_layers).to(self.device)
        self.v_back=scope.networks.UNetWithLinear(x_size=self.d,t_size=t_size,output_size=self.d,hidden_size=hiden_size,n_layers=n_layers).to(self.device)
        self.scale_m_fore=scope.networks.scale_model_muti(output_size=t_size).to(self.device)
        self.scale_m_back=scope.networks.scale_model_muti(output_size=t_size).to(self.device)


        self.backbone_trainer = scope.backbone_train.backbone_trainer(
            self.backbone_lr,self.N_pretraining,self.B,
            self.t_lists_stage,self.eps,
            ema=self.ema,decay=self.decay,
            early_stop=self.early_stop,patience=self.patience,
            lambda_=self.lambda_,
            plot_loss=self.plot_loss,save_model=self.save,
            save_path=self.save_path,
            record_gap=self.record_gap,
            prematched=self.prematched
        )
        self.fine_tuner = scope.fine_tune.fine_tuner(
            self.finetuning_lr,self.N_finetuning,self.B,
            self.t_lists_stage,
            self.delta_t,self.eps,
            ema=self.ema,decay=self.decay,
            lambda_=self.lambda_,
            plot_loss=self.plot_loss,save_model=self.save,
            save_path=self.save_path,
            record_gap=self.record_gap,
            prematched=self.prematched
        )

    def backbone_load(self,model_path='backbone.pt'):
        models_dict=torch.load(model_path)
        self.v_fore.load_state_dict(models_dict['v_fore'])
        self.v_back.load_state_dict(models_dict['v_back'])
        self.scale_m_fore.load_state_dict(models_dict['scale_m_fore'])
        self.scale_m_back.load_state_dict(models_dict['scale_m_back'])

        
    def finetuning_load(self,model_path='fine_tuned.pt'):
        models_dict=torch.load(model_path)
        self.v_fore_fine_tuned=copy.deepcopy(self.v_fore).train()
        self.v_back_fine_tuned=copy.deepcopy(self.v_back).train()
        self.scale_m_fore.load_state_dict(models_dict['scale_m_fore'])
        self.scale_m_back.load_state_dict(models_dict['scale_m_back'])
        self.v_fore_fine_tuned.load_state_dict(models_dict['v_fore_fine_tuned'])
        self.v_back_fine_tuned.load_state_dict(models_dict['v_back_fine_tuned'])


    def backbone_train(self):
        datasets = self.pi if not self.prematched else self.path_dataloader
        self.backbone_trainer.train(self.v_fore,self.v_back,self.scale_m_fore,self.scale_m_back,datasets)


    def fine_tune(self,change=10):
        self.v_fore_fine_tuned=copy.deepcopy(self.v_fore).train()
        self.v_back_fine_tuned=copy.deepcopy(self.v_back).train()
        datasets = self.pi if not self.prematched else self.path_dataloader
        self.fine_tuner.fine_tune(self.v_fore_fine_tuned,self.v_back_fine_tuned,self.scale_m_fore,self.scale_m_back,datasets,change=change)


    def eval_fore(self,test_0,v_fore):
        v_fore.eval()
        self.scale_m_fore.eval()
        with torch.no_grad():
            x_f=self.forward_generate(test_0,v_fore)
        return x_f

    def eval_back(self,test_1,v_back):
        v_back.eval()
        self.scale_m_back.eval()
        with torch.no_grad():
            x_b=self.backward_generate(test_1,v_back)
        return x_b


    def forward_generate(self,x_0,v_m):
        x_0.to(self.device)
        return scope.computations.generate_path(self.t_lists_stage,x_0,v_m,self.scale_m_fore,self.delta_t,self.eps)
        
        
    def backward_generate(self,x_1,v_m):
        x_1.to(self.device)
        return scope.computations.generate_path(self.t_lists_stage,x_1,v_m,self.scale_m_back,self.delta_t,self.eps,forward=False)
    
    
    def forward_generate_with_calibration(self,x_0,v_m,cluster_models,target_counts,max_attempts_per_branch=100,debug=False):
        v_m.eval()
        self.scale_m_fore.eval()
        with torch.no_grad():
            snapshots, (tracked_paths, final_surviving_path_ids, final_lineage) = md.calibrated_generation_with_debug(
                x_0,self.timepoints,
                scope.computations.generate_one_stage,
                self.t_lists_stage,
                v_m,self.scale_m_fore,
                self.delta_t,self.eps,
                cluster_models,
                target_counts,
                max_attempts_per_branch=max_attempts_per_branch,
                debug=debug
            )
        return snapshots, (tracked_paths, final_surviving_path_ids, final_lineage)
    
    
    