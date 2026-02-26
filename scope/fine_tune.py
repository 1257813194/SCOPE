import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import scope.computations


class fine_tuner(object):
    def __init__(self,lr,n_epochs,batch_size,timepoint_lists_for_each_stage,delta_t,eps,ema=False,decay=0.8,lambda_=0,plot_loss=True,save_model=True,save_path='model_history',record_gap=10,prematched=False):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.lr = lr
        self.n_epochs = n_epochs 
        self.batch_size = batch_size
        self.timepoint_lists_for_each_stage = timepoint_lists_for_each_stage
        self.delta_t = delta_t
        self.eps = eps
        
        self.ema = ema
        self.decay = decay
        self.lambda_ = lambda_
        
        self.loss_history={'loss':[],"loss_fore":[],"loss_back":[]}
        self.plot_loss = plot_loss
        self.save_model = save_model
        self.save_path = save_path
        self.record_gap = record_gap
        
        self.prematched = prematched


    def loss_plot(self):
        list1 = self.loss_history['loss']
        x = range(len(list1))
        plt.plot(x, list1, label='loss')
        plt.legend()
        plt.show()
        
    
    def model_save(self,v_fore,v_back,scale_m_fore,scale_m_back):
        models_dict = {
            'v_fore_fine_tuned': v_fore.state_dict(),
            'v_back_fine_tuned': v_back.state_dict(),
            'scale_m_fore': scale_m_fore.state_dict(),
            'scale_m_back': scale_m_back.state_dict(),
            'loss_history':self.loss_history
                    }
        torch.save(models_dict, self.save_path+'/fine_tuned.pt')
        
        
    def para_update(self,data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion,optimizer,forward_training=True,params_backup=None):
        
        future_stages = len(data_tuple)-stage_index-1
        past_stages = stage_index
        
        if forward_training:
            future_stage = np.random.randint(future_stages)
            x_1 = data_tuple[stage_index+future_stage+1].to(self.device).double()
            for i in range(future_stage):
                x_1 = scope.computations.generate_one_stage(self.timepoint_lists_for_each_stage,x_1,v_back,scale_m_back,self.delta_t,self.eps,stage_index+future_stage-i,forward=False)[-1].to(self.device)
            x_0_hat=scope.computations.generate_one_stage(self.timepoint_lists_for_each_stage,x_1,v_back,scale_m_back,self.delta_t,self.eps,stage_index,forward=False)[-1].to(self.device)
            x_t,t,t_ceil,t_floor=scope.computations.Interp_t(self.timepoint_lists_for_each_stage,self.eps,x_0_hat,x_1,stage_index)
            index = (t!=t_ceil).squeeze()
        else:
            past_stage = np.random.randint(past_stages+1)
            x_0 = data_tuple[stage_index-past_stage].to(self.device).double()
            for i in range(past_stage):
                x_0 = scope.computations.generate_one_stage(self.timepoint_lists_for_each_stage,x_0,v_fore,scale_m_fore,self.delta_t,self.eps,stage_index-past_stage+i,forward=True)[-1].to(self.device)
            x_1_hat=scope.computations.generate_one_stage(self.timepoint_lists_for_each_stage,x_0,v_fore,scale_m_fore,self.delta_t,self.eps,stage_index,forward=True)[-1].to(self.device)
            x_t,t,t_ceil,t_floor=scope.computations.Interp_t(self.timepoint_lists_for_each_stage,self.eps,x_0,x_1_hat,stage_index)
            index = (t!=t_floor).squeeze()
        
        t_floor=torch.from_numpy(np.repeat(t_floor,t[index,:].shape[0])).reshape(-1,1).to(self.device)
    
        if forward_training:
            x = v_fore(x_t[index,:],scale_m_fore(t[index,:],t_floor))
            loss=criterion(x,
                            (x_1[index,:]-x_t[index,:])/(t_ceil-t[index,:]).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x),dim=1))
        else:
            x = v_back(x_t[index,:],scale_m_back(t[index,:],t_floor))
            loss=criterion(x,
                            (x_0[index,:]-x_t[index,:])/(t[index,:]-t_floor).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x),dim=1))

        optimizer.zero_grad()
        loss.backward()
        if forward_training:
            torch.nn.utils.clip_grad_norm_(v_fore.parameters(), max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(v_back.parameters(), max_norm=1.0)
        optimizer.step()
        if self.ema:
            if forward_training:
                scope.computations.apply_ema_to_model(v_fore,params_backup,self.decay)
            else:
                scope.computations.apply_ema_to_model(v_back,params_backup,self.decay)
        return loss


    def fine_tune(self,v_fore,v_back,scale_m_fore,scale_m_back,datasets,change):

        scale_m_fore.eval()
        scale_m_back.eval()
        criterion_fore = nn.MSELoss()
        criterion_back = nn.MSELoss()
        optimizer_fore = optim.Adam(v_fore.parameters(), lr=self.lr)
        optimizer_back = optim.Adam(v_back.parameters(), lr=self.lr)
        
        if self.ema:
            fore_params = {name: param.clone().detach() for name, param in v_fore.named_parameters()}
            back_params = {name: param.clone().detach() for name, param in v_back.named_parameters()}
            for param in fore_params.values():
                param.requires_grad = False  
            for param in back_params.values():
                param.requires_grad = False
        with tqdm(total=self.n_epochs) as pbar:
            for n in range(self.n_epochs):
                epoch_loss = 0
                epoch_loss_fore = 0
                epoch_loss_back = 0
                
                if self.prematched:

                    dataloader = datasets
      
                    for data_tuple in dataloader:
                        data_tuple = data_tuple["sequences"]
                        data_tuple = data_tuple.transpose(0, 1)
                        num_all=[data.shape[0] for data in data_tuple]
                        if len(set(num_all))!=1:
                            break
                        for stage_index in range(len(data_tuple)-1):
                            if (n//change)%2==0:
                                v_fore.train()
                                v_back.eval()
                                if self.ema:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,optimizer_fore,forward_training=True,params_backup=fore_params)
                                else:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,optimizer_fore,forward_training=True)
                                epoch_loss += loss
                                epoch_loss_fore += loss
                                
                            else:
                                v_back.train()
                                v_fore.eval()
                                if self.ema:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_back,optimizer_back,forward_training=False,params_backup=back_params)
                                else:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_back,optimizer_back,forward_training=False)    
                                epoch_loss += loss
                                epoch_loss_back += loss    




                else: 
                    
                    dataloader_list=[DataLoader(dataset,shuffle=True,batch_size=self.batch_size) for dataset in datasets]
                    
                    for data_tuple in zip(*dataloader_list):
                        data_tuple = [d[0] for d in data_tuple]
                        num_all=[data.shape[0] for data in data_tuple]
                        if len(set(num_all))!=1:
                            break
                        for stage_index in range(len(data_tuple)-1):
                            if (n//change)%2==0:
                                v_fore.train()
                                v_back.eval()
                                if self.ema:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,optimizer_fore,forward_training=True,params_backup=fore_params)
                                else:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,optimizer_fore,forward_training=True)
                                epoch_loss += loss
                                epoch_loss_fore += loss
                                
                            else:
                                v_back.train()
                                v_fore.eval()
                                if self.ema:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_back,optimizer_back,forward_training=False,params_backup=back_params)
                                else:
                                    loss = self.para_update(data_tuple,stage_index,v_fore,v_back,scale_m_fore,scale_m_back,criterion_back,optimizer_back,forward_training=False)    
                                epoch_loss += loss
                                epoch_loss_back += loss                           
                        

                if (n+1) % self.record_gap == 0: 
                    with torch.no_grad():
                        self.loss_history['loss'].append(epoch_loss.item())
                        if (n//change)%2==0:
                            self.loss_history['loss_fore'].append(epoch_loss_fore.item())
                        else:
                            self.loss_history['loss_back'].append(epoch_loss_back.item())
                pbar.set_description('processed: %d' % (1 + n))
                if (n//change)%2==0:
                    pbar.set_postfix({'loss':epoch_loss.detach().cpu().numpy(),'loss_fore':epoch_loss_fore.detach().cpu().numpy(),})
                else:
                    pbar.set_postfix({'loss':epoch_loss.detach().cpu().numpy(),'loss_back':epoch_loss_back.detach().cpu().numpy(),})
                pbar.update(1)
                        
        if self.save_model:
            self.model_save(v_fore,v_back,scale_m_fore,scale_m_back)
        if self.plot_loss:
            self.loss_plot()

