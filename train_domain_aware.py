import os
import torch
from tqdm import tqdm
from model_filter import PrototypicalNetworkhead1
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from utils import *
import numpy as np
from datasets import *
import pickle
import random
from Meta_optimizer import *
import time
import copy
import logging
from Welford import Welford
from copy import deepcopy

datanames = ['Quickdraw', 'Aircraft', 'CUB', 'MiniImagenet', 'Omniglot', 'Plantae', 'Electronic', 'CIFARFS', 'Fungi', 'Necessities']



class SequentialMeta(object):
    def __init__(self,model, args=None):
        self.args = args
        self.model=model
        self.init_lr=args.lr
        self.hyper_lr = args.hyper_lr
        self.run_stat = Welford()
        self.patience = 5
        self.delta = 0.2
        self.freeze = False
        self.data_counter = {}
        self.best_score = {}
        self.data_stepdict = {}
        self.memory_rep = []

        self.patientstep = 100
        for name in datanames:
            self.data_stepdict[name] = 0
        for name in datanames:
            self.data_counter[name] = 0
        for name in datanames:
            self.best_score[name] = None

        self.update_lr(domain_id=0, lr=1e-3)
        self.meta_optim = Meta_Optimizer(self.optimizer, self.args.hyper_lr, self.args.device, self.args.clip_hyper, self.args.layer_filters)
        str_save = '_'.join(datanames)
        self.step = 0
        self.ELBO = 0.0
        
        self.filepath = os.path.join(self.args.output_folder, 'protonet_Meta_Optimizer{}'.format(str_save), 'Block{}'.format(self.args.num_block), 'shot{}'.format(self.args.num_shot), 'way{}'.format(self.args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        

    def train(self, Interval, dataloader_dict, domain_id = None):
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
                    train_embeddings = self.model(train_inputs, domain_id)
                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                    test_embeddings = self.model(test_inputs, domain_id)

                    prototypes = get_prototypes(train_embeddings, train_targets, args.num_way)
                    loss = prototypical_loss(prototypes, test_embeddings, test_targets)
                    loss.backward(retain_graph=True)


                    #Reservoir sampling
                    if self.step < self.args.memory_limit:
                        savedict = batch
                        self.memory_rep.append(savedict)

                    else:
                        randind = random.randint(0, self.step)
                        if randind < self.args.memory_limit:
                            savedict =  batch
                            self.memory_rep[randind] = savedict

                    self.step = self.step+1


                    grad_list = []
                    param_names = []
                    for name, v in self.model.named_parameters():
                        if 'domain_out' not in name: 
                            if v.requires_grad:
                                grad_list.append(v.grad)
                                param_names.append(name)

                    first_grad = grad_list
                    
                    count = self.args.sample
                    if self.memory_rep:
                        num_memory = len(self.memory_rep)
                        if num_memory<count:
                            selectmemory = self.memory_rep
                        else:
                            samplelist = random.sample(range(num_memory), count)
                            selectmemory = []
                            for ind in samplelist:
                                selectmemory.append(self.memory_rep[ind])

                    # Dynamical freeze mechanism

                    
                    if  self.memory_rep:
                        memory_dict, summemory_loss = rep_memory_dict(self.args, self.model, selectmemory)
                        loss += summemory_loss

                        memory_loss = 0.0
                        for key in memory_dict:
                            memory_loss += memory_dict[key] 

                        flat = []
                        for name, param in self.model.named_parameters():
                                flat.append(param.view(-1))

                        flat = torch.cat(flat)
                        flat_np = flat.cpu().data.numpy()
                        self.run_stat(flat_np)

                        if  self.data_stepdict[dataname] > 0:
                            logprob = loss.item() 
                            memory_loss /= len(memory_dict)
                            logprob += memory_loss
                            count = self.data_stepdict[dataname]%30
                            self.ELBO = self.ELBO +(logprob-self.ELBO)/count
                            
                        if  self.data_stepdict[dataname] > 0 and self.data_stepdict[dataname]%30 ==0:
                            

                            self.ELBO -= math.log2(np.sum(self.run_stat.std))
                            self.run_stat = Welford()
                            self.ELBO = 0.0

                            if  self.data_stepdict[dataname] > self.patientstep:
                                if self.freeze == False:
                                    if self.best_score[dataname] is None:
                                        self.best_score[dataname] = ELBO
                                    elif ELBO > self.best_score[dataname] + self.delta:
                                        self.data_counter[dataname] = self.data_counter[dataname] + 1
                                        
                                        if self.data_counter[dataname] >= self.patience:
                                            self.freeze = True
                                            description = 'Interval_{}_EarlyStopping counter dataname {}: {} out of {}'.format(Interval, dataname, self.data_counter[dataname], self.patience)
                                            print('description', description)
                                            self.update_lr(domain_id, lr=0.0)
                                    else:
                                        self.best_score[dataname] = ELBO
                            else:
                                self.freeze = False

                    
                    
                    val_graddict = {}
                    layer_name = []
                    for gradient, name in zip(first_grad, param_names):
                        split_name = name.split('.')
                        layer = split_name[0]
                        if layer not in self.args.layer_filters:
                            if layer not in layer_name:
                                layer_name.append(layer)
                                val_graddict[layer] = []
                                val_graddict[layer].append(gradient.clone().view(-1))
                            else:
                                val_graddict[layer].append(gradient.clone().view(-1))
                        else:
                            layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                            if layer_sub not in layer_name:
                                layer_name.append(layer_sub)
                                val_graddict[layer_sub] = []
                                val_graddict[layer_sub].append(gradient.clone().view(-1))
                            else:
                                val_graddict[layer_sub].append(gradient.clone().view(-1))

                    for key in val_graddict:
                        val_graddict[key] = torch.cat(val_graddict[key])
                    self.optimizer.step()
                    
                    if self.memory_rep:
                        
                        self.meta_optim.optimizer = self.optimizer
                        self.meta_optim.meta_gradient(self.model, val_graddict)

                        count = self.args.sample
                        num_memory = len(self.memory_rep)
                        if num_memory<count:
                            selectmemory = self.memory_rep
                        else:
                            samplelist = random.sample(range(num_memory), count)
                            selectmemory = []
                            for ind in samplelist:
                                selectmemory.append(self.memory_rep[ind])


                        val_grad = self.rep_grad_new(self.args, selectmemory)
                        
                        self.meta_optim.meta_step(val_grad)
                        self.model.zero_grad()
                    

                    if batch_idx >= args.num_batches:
                        break


    def rep_grad_new(self, args, selectmemory):


        memory_loss =0
        for dataidx, select in enumerate(selectmemory):

                    memory_train_inputs, memory_train_targets = select['train'] 
                    memory_train_inputs = memory_train_inputs.to(device=args.device)
                    memory_train_targets = memory_train_targets.to(device=args.device)
                    if memory_train_inputs.size(2) == 1:
                        memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
                    memory_train_embeddings = self.model(memory_train_inputs, dataidx)

                    memory_test_inputs, memory_test_targets = select['test'] 
                    memory_test_inputs = memory_test_inputs.to(device=args.device)
                    memory_test_targets = memory_test_targets.to(device=args.device)
                    if memory_test_inputs.size(2) == 1:
                        memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)
                   
                    memory_test_embeddings = self.model(memory_test_inputs, dataidx)
                    memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
                    memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

        
        param_list = []
        param_names = []
        for name, v in self.model.named_parameters():
            if 'domain_out' not in name: 
                if v.requires_grad:
                    param_list.append(v)
                    param_names.append(name)
        val_grad = torch.autograd.grad(memory_loss, param_list)
        
        val_graddict = {}
        layer_name = []
        for gradient, name in zip(val_grad, param_names):
            split_name = name.split('.')
            layer = split_name[0]
            if layer not in self.args.layer_filters:
                if layer not in layer_name:
                    layer_name.append(layer)
                    val_graddict[layer] = []
                    val_graddict[layer].append(gradient.view(-1))
                else:
                    val_graddict[layer].append(gradient.view(-1))
            else:
                layer_sub = layer+'.'+split_name[1]+'.'+split_name[2]
                if layer_sub not in layer_name:
                    layer_name.append(layer_sub)
                    val_graddict[layer_sub] = []
                    val_graddict[layer_sub].append(gradient.view(-1))
                else:
                    val_graddict[layer_sub].append(gradient.view(-1))

        for key in val_graddict:
            val_graddict[key] = torch.cat(val_graddict[key])
        self.model.zero_grad()
        memory_loss.detach_()
        return val_graddict

    
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
 

    def valid(self, dataloader_dict, domain_id, Interval):
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

                        train_embeddings = self.model(train_inputs, domain_id)
                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings = self.model(test_inputs, domain_id)

                        prototypes = get_prototypes(train_embeddings, train_targets, self.args.num_way)
                        accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                        acc_list.append(accuracy.cpu().data.numpy())
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break

            avg_accuracy = np.round(np.mean(acc_list), 4)
            acc_dict = {dataname:avg_accuracy}
            logging.debug('Interval_{}_{}_accuracy_{}'.format(Interval, dataname, avg_accuracy))

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
    model = PrototypicalNetworkhead1(3,
                                args.embedding_size,
                                hidden_size=args.hidden_size, num_tasks=len(datanames), num_block = args.num_block)
    model.to(device=args.device)

    num_data = len(train_loader_list)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

    each_Interval = args.num_Interval
    seqmeta = SequentialMeta(model, args=args)


    domain_acc = []
    for loaderindex, train_loader in enumerate(train_loader_list):
        model.set_req_grad(loaderindex, False)
        seqmeta.update_lr(loaderindex, lr = args.lr)
        for Interval in range(each_Interval*loaderindex, each_Interval*(loaderindex+1)):
            print('Interval {}'.format(Interval))
            seqmeta.train(Interval, train_loader, domain_id = loaderindex)

            total_acc = 0.0
            Interval_acc = []
            for index, test_loader in enumerate(test_loader_list[:loaderindex+1]):
                test_accuracy_dict = seqmeta.valid(test_loader, domain_id = index, Interval = Interval)
                Interval_acc.append(test_accuracy_dict)
                acc = list(test_accuracy_dict.values())[0]
                total_acc += acc

                if Interval == (each_Interval*(loaderindex+1)-1) and index == loaderindex:
                    domain_acc.append(test_accuracy_dict)
            avg_acc = total_acc/(loaderindex+1)
            print('average testing accuracy', avg_acc)

            
            all_accdict[str(Interval)] = Interval_acc
            with open(seqmeta.filepath + '/stats_acc.pickle', 'wb') as handle:
                pickle.dump(all_accdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if loaderindex>0:
            BWT = 0.0            
            for index, (best_domain, Interval_domain) in enumerate(zip(domain_acc, Interval_acc)):
                best_acc = list(best_domain.values())[0]
                each_acc = list(Interval_domain.values())[0]
                BWT += each_acc - best_acc
            avg_BWT = BWT/index
            print('avg_BWT', avg_BWT)

        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Sequential domain meta learning')
    parser.add_argument('--data_path', type=str, default='/data/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--output_folder', type=str, default='output/CVPR/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--num-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')
    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for each domain (default: 4).')
    

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
        help='Number of batches the model is tested over (default: 150).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 10).')
    parser.add_argument('--sample', type=int, default=1,
        help='Number of memory tasks per iteration.')
    parser.add_argument('--memory_limit', type=int, default=10,
        help='Number of batches in the memory buffer.')        
    parser.add_argument('--num_Interval', type=int, default=25,
        help='Number of Intervals for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for testing (default: 5).')
    parser.add_argument('--lr', type=float, default=1e-3,
        help='learning rate.')
    parser.add_argument('--clip_hyper', type=float, default=10.0)
    parser.add_argument('--hyper-lr', type=float, default=1e-4)
    parser.add_argument('--layer_filters', type=int, nargs='+', default=['conv1', 'conv2', 'conv3', 'conv4'], help='0 = CPU.')
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    main(args)
