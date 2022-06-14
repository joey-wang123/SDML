import math
import numpy as np
import torch.optim as optim
import sys
import time
import torch

class Meta_Optimizer(object):

    def __init__(self, optimizer, hyper_lr, device, clip_hyper, layer_filters):
        self.optimizer = optimizer
        self.clip_hyper = clip_hyper
        self.beta1 = self.optimizer.param_groups[0]['betas'][0]
        self.beta2 = self.optimizer.param_groups[0]['betas'][1]
        self.eps = self.optimizer.param_groups[0]['eps']
        self.hyper_lr = hyper_lr

        self.lr_adapt = []
        for ind in range(len(self.optimizer.param_groups)-1):
            lr_layer = torch.tensor(self.optimizer.param_groups[ind]['lr'], requires_grad=True, device=device)
            self.lr_adapt.append(lr_layer)
        self.lr_optim = optim.SGD(self.lr_adapt, lr=self.hyper_lr)
        self.device = device
        self.layer_filters = layer_filters
        self.z = None
        self.step = 0
        self.state_init = False

    def adapt(self, net):
        exp_avg_dict = {}
        exp_avg_sq_dict = {}

        layer_name = []
        for name, v in net.named_parameters():
            if 'domain_out' not in name:
                    if v.requires_grad:
                        split_name = name.split('.')
                        layer = split_name[0]
                        sublayer = split_name[1]
                        if layer not in self.layer_filters:
                            if layer not in layer_name:
                                layer_name.append(layer)
                                exp_avg_dict[layer] = []
                                exp_avg_sq_dict[layer] = []
                                exp_avg_dict[layer].append(self.optimizer.state[v]['exp_avg'].view(-1)) 
                                exp_avg_sq_dict[layer].append(self.optimizer.state[v]['exp_avg_sq'].view(-1)) 
                            else:
                                exp_avg_dict[layer].append(self.optimizer.state[v]['exp_avg'].view(-1)) 
                                exp_avg_sq_dict[layer].append(self.optimizer.state[v]['exp_avg_sq'].view(-1)) 

                        else:
                            layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                            if layer_sub not in layer_name:
                                layer_name.append(layer_sub)
                                exp_avg_dict[layer_sub] = []
                                exp_avg_sq_dict[layer_sub] = []
                                exp_avg_dict[layer_sub].append(self.optimizer.state[v]['exp_avg'].view(-1)) 
                                exp_avg_sq_dict[layer_sub].append(self.optimizer.state[v]['exp_avg_sq'].view(-1))
                            else: 
                                exp_avg_dict[layer_sub].append(self.optimizer.state[v]['exp_avg'].view(-1)) 
                                exp_avg_sq_dict[layer_sub].append(self.optimizer.state[v]['exp_avg_sq'].view(-1))
            
        for key in exp_avg_dict:
            exp_avg_dict[key] = torch.cat(exp_avg_dict[key])
        for key in exp_avg_sq_dict:
            exp_avg_sq_dict[key] = torch.cat(exp_avg_sq_dict[key])

        return (exp_avg_dict, exp_avg_sq_dict)


    def meta_gradient(self, net, first_grad):
            if not self.state_init:
                self.state_init = True
                self.step += 1
                return

            coeff = (math.sqrt(1.0 - self.beta2 ** self.step)) / (1.0 - self.beta1 ** self.step)
            if self.z is None:
                m_t, v_t = self.adapt(net)
                z = {}
                for key in m_t:
                    z[key] = torch.neg(coeff * (m_t[key] / torch.sqrt(v_t[key] + self.eps)))
                self.z = z
            self.step += 1

            for key in self.z:
                self.z[key] = self.z[key].detach()



    def meta_step(self, val_grad):

        if self.z is None:
            return

        lr_grad = {}
        for key in val_grad:
            lr_grad[key] = val_grad[key] @ self.z[key]

        for key, layer_lr in zip(lr_grad, self.lr_adapt):
            layer_lr.grad = lr_grad[key].clamp_(-self.clip_hyper, self.clip_hyper).clone().detach()+1.0
        
        self.lr_optim.step()
        new_lr = {}
        for (key, layer_lr) in zip(val_grad, self.lr_adapt):
            new_lr[key] = layer_lr.data.clamp_(0, 1e-5).data.item()
            
        for index, (key, param_group) in enumerate(zip(new_lr, self.optimizer.param_groups)):
            param_group['lr'] = new_lr[key]


