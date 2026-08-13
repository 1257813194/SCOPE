import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

import scope.computations


class backbone_trainer(object):
    def __init__(self,lr,n_epochs,batch_size,timepoint_lists_for_each_stage,eps,ema=False,decay=0.8,early_stop=False,patience=10,lambda_=0,plot_loss=True,save_model=True,save_path='model_history',record_gap=10,prematched=False):
        
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        
        self.lr = lr
        self.n_epochs = n_epochs 
        self.batch_size = batch_size
        self.eps = eps
        
        self.ema = ema
        self.decay = decay
        self.early_stop = early_stop
        self.patience = patience
        self.lambda_ = lambda_
        self.timepoint_lists_for_each_stage = timepoint_lists_for_each_stage
        
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
            'v_fore': v_fore.state_dict(),
            'v_back': v_back.state_dict(),
            'scale_m_fore': scale_m_fore.state_dict(),
            'scale_m_back': scale_m_back.state_dict(),
            'loss_history':self.loss_history
                    }
        torch.save(models_dict, self.save_path+'/backbone.pt')
        

    def train_an_epoch(self,v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,criterion_back,optimizer,dataloader_list,v_fore_params=None,v_back_params=None):
        epoch_loss=0
        epoch_loss_fore = 0
        epoch_loss_back = 0
        for data_tuple in zip(*dataloader_list):
            data_tuple = [d[0] for d in data_tuple]
            num_all=[data.shape[0] for data in data_tuple]
            if len(set(num_all))!=1:
                break
            for stage in range(len(data_tuple)-1):
                x_0=data_tuple[stage].to(self.device)
                x_1=data_tuple[stage+1].to(self.device)
                x_t,t,t_ceil,t_floor=scope.computations.Interp_t(self.timepoint_lists_for_each_stage,self.eps,x_0,x_1,stage,bondary_constrain=True)
                batch_size=x_0.shape[0]
                b=batch_size//2
                t_fore = t[:b]
                t_back = t[b:]
                x_t_fore = x_t[:b]
                x_t_back = x_t[b:]
                t_floor = torch.from_numpy(np.repeat(t_floor,t_fore.shape[0])).reshape(-1,1).to(self.device)
                y_t_fore=scale_m_fore(t_fore,t_floor)
                y_t_back=scale_m_back(t_back,t_floor)
                x_fore=v_fore(x_t_fore,y_t_fore)
                x_back=v_back(x_t_back,y_t_back)
                loss_fore=criterion_fore(x_fore,(x_1[:b]-x_t_fore)/(t_ceil-t_fore).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x_fore),dim=1))
                loss_back=criterion_back(x_back,(x_0[b:]-x_t_back)/(t_back-t_floor).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x_back),dim=1))
                loss=0.5*(loss_fore+loss_back)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(v_fore.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(v_back.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(scale_m_fore.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(scale_m_back.parameters(), max_norm=1.0)
                optimizer.step()
                if self.ema:
                    scope.computations.apply_ema_to_model(v_fore,v_fore_params,self.decay)
                    scope.computations.apply_ema_to_model(v_back,v_back_params,self.decay)
                epoch_loss += loss
                epoch_loss_fore += loss_fore
                epoch_loss_back += loss_back
        return epoch_loss, epoch_loss_fore, epoch_loss_back




    def train_an_epoch_prematched(self,v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,criterion_back,optimizer,dataloader,v_fore_params=None,v_back_params=None):
        epoch_loss=0
        epoch_loss_fore = 0
        epoch_loss_back = 0
        for data_tuple in dataloader:
            data_tuple = data_tuple["sequences"]
            data_tuple = data_tuple.transpose(0, 1)
            for stage in range(len(data_tuple)-1):
                x_0=data_tuple[stage].to(self.device)
                x_1=data_tuple[stage+1].to(self.device)
                x_t,t,t_ceil,t_floor=scope.computations.Interp_t(self.timepoint_lists_for_each_stage,self.eps,x_0,x_1,stage,bondary_constrain=True)
                batch_size=x_0.shape[0]
                b=batch_size//2
                t_fore = t[:b]
                t_back = t[b:]
                x_t_fore = x_t[:b]
                x_t_back = x_t[b:]
                t_floor = torch.from_numpy(np.repeat(t_floor,t_fore.shape[0])).reshape(-1,1).to(self.device)
                y_t_fore=scale_m_fore(t_fore,t_floor)
                y_t_back=scale_m_back(t_back,t_floor)
                x_fore=v_fore(x_t_fore,y_t_fore)
                x_back=v_back(x_t_back,y_t_back)
                loss_fore=criterion_fore(x_fore,(x_1[:b]-x_t_fore)/(t_ceil-t_fore).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x_fore),dim=1))
                loss_back=criterion_back(x_back,(x_0[b:]-x_t_back)/(t_back-t_floor).view(-1,1)) + self.lambda_ * torch.mean(torch.sum(torch.abs(x_back),dim=1))
                loss=0.5*(loss_fore+loss_back)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(v_fore.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(v_back.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(scale_m_fore.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(scale_m_back.parameters(), max_norm=1.0)
                optimizer.step()
                if self.ema:
                    scope.computations.apply_ema_to_model(v_fore,v_fore_params,self.decay)
                    scope.computations.apply_ema_to_model(v_back,v_back_params,self.decay)
                epoch_loss += loss
                epoch_loss_fore += loss_fore
                epoch_loss_back += loss_back
        return epoch_loss, epoch_loss_fore, epoch_loss_back







    def train(self,v_fore,v_back,scale_m_fore,scale_m_back,datasets):
        
        criterion_fore = nn.MSELoss()
        criterion_back = nn.MSELoss()
        optimizer = optim.Adam(list(v_fore.parameters())+list(v_back.parameters())+list(scale_m_fore.parameters())+list(scale_m_back.parameters()), lr=self.lr)
        
        if self.ema:
            v_fore_params = {name: param.clone().detach() for name, param in v_fore.named_parameters()}
            v_back_params = {name: param.clone().detach() for name, param in v_back.named_parameters()}
            for param in v_fore_params.values():
                param.requires_grad = False  
            for param in v_back_params.values():
                param.requires_grad = False  

        if self.early_stop:
            best_loss = np.inf
            epochs_no_improve = 0
            stop_indicator = False
        
        with tqdm(total=self.n_epochs) as pbar:
            for n in range(self.n_epochs):
                if self.early_stop:
                    if stop_indicator:
                        print("Early stopping at epoch", n)
                        break
                
                if not self.prematched:
                    dataloader_list=[DataLoader(dataset,shuffle=True,batch_size=self.batch_size) for dataset in datasets]
      
                    if self.ema:
                        epoch_loss, epoch_loss_fore, epoch_loss_back = self.train_an_epoch(v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,criterion_back,optimizer,dataloader_list,v_fore_params,v_back_params)
                    else:
                        epoch_loss, epoch_loss_fore, epoch_loss_back = self.train_an_epoch(v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,criterion_back,optimizer,dataloader_list)
                
                else:
                    
                    dataloader = datasets
      
                    if self.ema:
                        epoch_loss, epoch_loss_fore, epoch_loss_back = self.train_an_epoch_prematched(v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,criterion_back,optimizer,dataloader,v_fore_params,v_back_params)
                    else:
                        epoch_loss, epoch_loss_fore, epoch_loss_back = self.train_an_epoch_prematched(v_fore,v_back,scale_m_fore,scale_m_back,criterion_fore,criterion_back,optimizer,dataloader)    


                
                if (n+1) % self.record_gap == 0: 
                    with torch.no_grad():
                        self.loss_history['loss'].append(epoch_loss.item())
                        self.loss_history['loss_fore'].append(epoch_loss_fore.item())
                        self.loss_history['loss_back'].append(epoch_loss_back.item())
                pbar.set_description('processed: %d' % (1 + n))
                pbar.set_postfix({'loss':epoch_loss.detach().cpu().numpy(),'loss_fore':epoch_loss_fore.detach().cpu().numpy(),'loss_back':epoch_loss_back.detach().cpu().numpy(),})
                pbar.update(1)
                
                if self.early_stop:                
                    if epoch_loss.item() < best_loss:
                        best_loss = epoch_loss.item()
                        epochs_no_improve = 0  
                    else:
                        epochs_no_improve += 1
                    if epochs_no_improve >= self.patience:
                        stop_indicator = True
                        
        if self.save_model:
            self.model_save(v_fore,v_back,scale_m_fore,scale_m_back)
        if self.plot_loss:
            self.loss_plot()

