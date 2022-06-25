#some codes are borrowed from https://github.com/tristandeleu/pytorch-meta/tree/master/examples/protonet

import os
import torch
from tqdm import tqdm
from model_filter import PrototypicalNetworkinfer
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import get_accuracy, get_accuracy_pred
import numpy as np
from datasets import *
import pickle
import random
from Meta_optimizer import *
import time
import copy
import logging
import changepointdetection.python.pythonmultivariate.StudentTMulti as st
import changepointdetection.python.pythonmultivariate.Detector as dt
import changepointdetection.python.pythonmultivariate.hazards as hz
from functools import partial
import warnings
warnings.filterwarnings("ignore")


datanames = ['Quickdraw', 'Aircraft', 'CUB', 'MiniImagenet', 'Omniglot', 'Plantae', 'Electronic', 'CIFARFS', 'Fungi', 'Necessities']


class SequentialMeta(object):
    def __init__(self,model, lr=0.001, args=None):
        self.args = args
        self.model=model
        self.init_lr=lr
        self.hyper_lr = args.hyper_lr
        self.update_lr(domain_id=0, lr=1e-3)
        self.hyper_optim = Meta_Optimizer(self.optimizer, self.args.hyper_lr, self.args.device, self.args.clip_hyper, self.args.layer_filters)
        str_save = '_'.join(datanames)
        self.step = 0
        self.domain_id  = 0
        self.window = []
        self.estimate_id = 0
        self.domain_iter = {}
        self.domain_iter['0'] = 0
        self.domain_embed = {}
        for ind in range(len(datanames)+5):
            self.domain_embed[str(ind)] = 0.0
        self.memory_rep = []
        self.countind = 0

        for ind in range(6):
            self.domain_iter[str(ind)] = 0

        self.numsteps = 2
        mean = 0.0
        self.detector = dt.Detector()
        self.prior = st.StudentTMulti(self.numsteps, mean)
        self.interval = 700
        self.startiter = 800
        if self.startiter> self.interval:
            self.countind = self.startiter // self.interval
        else:
            self.countind = 0


        print('self.countpoint', self.countind)
        self.countpoint = 0
        self.filepath = os.path.join(self.args.output_folder, 'protonet_changepoint2_{}_Embed_dim_{}_windowsteps_{}'.format(str_save, args.embedding_size, self.numsteps), 'Block{}'.format(self.args.num_block), 'shot{}'.format(self.args.num_shot), 'way{}'.format(self.args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)


    def train(self, Interval, dataloader_dict, domain_id, new, memory_train = None):

        self.model.train()
        for dataname, dataloader in dataloader_dict.items():
            with tqdm(dataloader, total=self.args.num_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.model.zero_grad()
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    if train_inputs.size(2) == 1:
                        train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                    train_embeddings = self.model(train_inputs, self.domain_id)[0]

                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                    test_embeddings = self.model(test_inputs, self.domain_id)[0]

                    prototypes = get_prototypes(train_embeddings, train_targets, args.num_way)
                    loss = prototypical_loss(prototypes, test_embeddings, test_targets)
                    logfile = True
                    if logfile:
                        alpha = 0.5
                        mean_proto = torch.mean(prototypes, dim = (0,1))
                        self.domain_embed[str(self.domain_id)] = alpha*torch.mean(test_embeddings, dim= (0,1)) + (1-alpha)*self.domain_embed[str(self.domain_id)]
                        lenwindow = 20
                        if (len(self.window))> lenwindow:
                            self.window.remove(self.window[0])
                        
                        if self.window:
                            if (len(self.window)) == lenwindow:
                                dist_list = []
                                for proto in self.window[-1*(self.numsteps+1):-1]:
                                    currentdist = torch.sum((mean_proto - proto)**2).item()
                                    dist_list.append(currentdist)


                            if self.step>self.startiter:
                                if self.step % self.interval == 0:
                                        self.countind += 1
                                        dim =  self.numsteps
                                        mean = 0.0 
                                        self.detector = dt.Detector()
                                        self.prior = st.StudentTMulti(dim, mean)
                                        self.countpoint = 0
                                x = torch.tensor(dist_list).cpu().detach().numpy()
                                self.detector.detect(x,partial(hz.constant_hazard,lam=200),self.prior)


                                prev = copy.deepcopy(self.countpoint)
                                maxes, CP, theta = self.detector.retrieve(self.prior)
                                self.countpoint = len(CP)


                                if self.countpoint>1 and (self.countpoint-prev)>0 and self.domain_iter[str(self.domain_id)]>500:
                                    self.domain_embed[str(self.domain_id)] = self.window[0]
                                    self.estimate_id +=1
                                    self.domain_iter[str(self.estimate_id)] = 0
                                    self.domain_id = self.estimate_id


                                    self.model.set_req_grad(self.domain_id, False)
                                    self.update_lr(self.domain_id, lr=1e-3)
                                    self.domain_embed[str(self.domain_id)] = self.window[-1]                 
 
                                self.domain_iter[str(self.domain_id)]+= 1


                        self.window.append(self.domain_embed[str(self.domain_id)]) 

                    if self.step < self.args.memory_limit:
                        self.memory_rep.append(batch)

                    else:
                        randind = random.randint(0, self.step)
                        if randind < self.args.memory_limit:
                            self.memory_rep[randind] = batch

                    loss.backward()
                    self.optimizer.step()

                    self.step = self.step +1
                    if batch_idx >= args.num_batches:
                        break


    def save(self, Interval):
        if self.args.output_folder is not None:
            filename = os.path.join(self.filepath, 'Interval{0}.pt'.format(Interval))
            with open(filename, 'wb') as f:
                state_dict = self.model.state_dict()
                torch.save(state_dict, f)



    def load(self, Interval):
        filename = os.path.join(self.filepath,  'Interval{0}.pt'.format(Interval))
        print('loading model filename', filename)
        self.model.load_state_dict(torch.load(filename))
 
    def valid_test(self, dataloader_dict, domain_id, Interval):
        self.model.eval()
        acc_dict = {}
        acc_list = []
        for dataname, dataloader in dataloader_dict.items():
            with torch.no_grad():
                with tqdm(dataloader, total=self.args.num_valid_batches) as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        self.model.zero_grad()
                        train_inputs, train_targets = batch['train']
                        train_inputs = train_inputs.to(device=self.args.device)
                        train_targets = train_targets.to(device=self.args.device)
                        if train_inputs.size(2) == 1:
                            train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)

                        train_embeddings = self.model(train_inputs, domain_id)[0]
                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings = self.model(test_inputs, domain_id)[0]

                        prototypes = get_prototypes(train_embeddings, train_targets, self.args.num_way)
                        accuracy, sq_distances, predictions = get_accuracy_pred(prototypes, test_embeddings, test_targets)
                        acc_list.append(accuracy.cpu().data.numpy())
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break

            avg_accuracy = np.round(np.mean(acc_list), 4)
            acc_dict = {dataname:avg_accuracy}


            return acc_dict
       


    def update_lr(self, domain_id, lr=None):
        params_dict = []
        if domain_id==0:
            layer_params = {}
            layer_name = []
            fast_parameters = []
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    if 'conv' in name:
                        split_name = name.split('.')
                        layer = split_name[0]
                        if layer not in self.args.layer_filters:
                            if layer not in layer_name:
                                layer_name.append(layer)
                                layer_params[layer] = []
                                layer_params[layer].append(p)
                            else:
                                layer_params[layer].append(p)

                        else:
                            layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                            if layer_sub not in layer_name:
                                layer_name.append(layer_sub)
                                layer_params[layer_sub] = []
                                layer_params[layer_sub].append(p)
                            else:
                                layer_params[layer_sub].append(p)

                    else:
                        fast_parameters.append(p)

            params_list = []
            for key in layer_params:
                params_list.append({'params':layer_params[key], 'lr':self.init_lr})
            params_list.append({'params':fast_parameters, 'lr':self.init_lr})
            self.optimizer = torch.optim.Adam(params_list, lr=self.init_lr)
        else:
            layer_params = {}
            layer_name = []
            fast_parameters = []
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    if 'conv' in name:
                        split_name = name.split('.')
                        layer = split_name[0]
                        if layer not in self.args.layer_filters:
                            if layer not in layer_name:
                                layer_name.append(layer)
                                layer_params[layer] = []
                                layer_params[layer].append(p)
                            else:
                                layer_params[layer].append(p)

                        else:
                            layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                            if layer_sub not in layer_name:
                                layer_name.append(layer_sub)
                                layer_params[layer_sub] = []
                                layer_params[layer_sub].append(p)
                            else:
                                layer_params[layer_sub].append(p)
                    else:
                        fast_parameters.append(p)

            params_list = []
            for key in layer_params:
                params_list.append({'params':layer_params[key], 'lr':lr})
            params_list.append({'params':fast_parameters, 'lr':self.init_lr})
            self.optimizer = torch.optim.Adam(params_list, lr=self.init_lr)

    



def main(args):

    all_accdict = {}
    train_loader_list, valid_loader_list, test_loader_list = dataset(args, datanames)
    model = PrototypicalNetworkinfer(3,
                                args.embedding_size,
                                hidden_size=args.hidden_size, num_tasks=len(datanames)+15, num_block = args.num_block)
    model.to(device=args.device)
    

    num_data = len(train_loader_list)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    each_Interval = args.num_Interval
    seqmeta = SequentialMeta(model, args=args)
    seqmeta.update_lr(0, lr=1e-3)

    seqmeta.domain_id =  0
    seqmeta.estimate_id = 0


    for loaderindex, train_loader in enumerate(train_loader_list):
        for Interval in range(each_Interval*loaderindex, each_Interval*(loaderindex+1)):
            print('Interval {}'.format(Interval))

            memory_train = None
            train_domainid = loaderindex
            
            if Interval == each_Interval*loaderindex:
                new = True
            else:
                new = False
            seqmeta.train(Interval, train_loader, train_domainid, new, memory_train)
            Interval_acc = []
            test_loader = test_loader_list[loaderindex]
            test_accuracy_dict = seqmeta.valid_test(test_loader, domain_id = seqmeta.domain_id, Interval = Interval)
            Interval_acc.append(test_accuracy_dict)

            all_accdict[str(Interval)] = Interval_acc 
            print('seqmeta.domain_id', seqmeta.domain_id)
            with open(seqmeta.filepath + '/stats_acc.pickle', 'wb') as handle:
                pickle.dump(all_accdict, handle, protocol=pickle.HIGHEST_PROTOCOL)


 




if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Sequential domain meta learning')
    parser.add_argument('--data_path', type=str, default='/media/zheshiyige/Elements/NIPS2021 data/data/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output_folder', type=str, default='output/BOCPD/',
        help='Path to the output folder for saving the model (optional).')

    parser.add_argument('--MiniImagenet_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for MiniImagenet (default: 4).')
    parser.add_argument('--CIFARFS_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for CIFARFS (default: 4).')
    parser.add_argument('--CUB_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for CUB (default: 4).')
    parser.add_argument('--Aircraft_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Omniglot_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Omniglot (default: 4).')
    parser.add_argument('--Plantae_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Aircraft (default: 4).')
    parser.add_argument('--Quickdraw_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Quickdraw (default: 4).')
    parser.add_argument('--VGGflower_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for VGGflower (default: 4).')
    parser.add_argument('--Fungi_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Fungiflower (default: 4).')
    parser.add_argument('--Logo_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Logo (default: 4).')
    parser.add_argument('--num_block', type=int, default=4,
        help='Number of convolution block.')
    parser.add_argument('--num-batches', type=int, default=200,
        help='Number of batches the prototypical network is trained over (default: 200).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num_memory_batches', type=int, default=1,
        help='Number of batches the model is trained over (default: 1).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 10).')
    parser.add_argument('--num_Interval', type=int, default=25,
        help='Number of Intervals for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for validation (default: 5).')
    parser.add_argument('--memory_limit', type=int, default=100,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--clip_hyper', type=float, default=10.0)
    parser.add_argument('--LR', type=float, default=2.0)
    parser.add_argument('--hyper-lr', type=float, default=1e-4)
    parser.add_argument('--layer_filters', type=int, nargs='+', default=['conv1', 'conv2', 'conv3', 'conv4'], help='layerfilters')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    args = parser.parse_args()

    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device)
    print('args.device', args.device)
    main(args)

