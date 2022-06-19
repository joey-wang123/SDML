import os
import torch
from tqdm import tqdm
import logging
from torchmeta.datasets import Omniglot, MiniImagenet, CIFARFS, CUB
from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from torchmeta.transforms import Categorical, ClassSplitter
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from model import PrototypicalNetwork
from utils import get_accuracy
import numpy as np
from datasets import *
import pickle
import random
from srcbayes.networks import net_ucb as network
from srcbayes.approaches.utils import BayesianSGD, BayesianAdam
import copy


datanames = ['Quickdraw', 'Aircraft', 'CUB', 'MiniImagenet', 'Omniglot', 'Plantae', 'Electronic', 'CIFARFS', 'Fungi', 'Necessities']

class PNetUCB(object):
    def __init__(self,model,optimizer, nIntervals=100,sbatch=64,lr=0.05,lr_min=1e-4,lr_factor=3,lr_patience=5,clipgrad=10000,lamb=0.75,smax=400,args=None):
        self.args = args
        self.model=model
        self.lr_min=lr_min
        self.lr_factor=lr_factor
        self.lr_patience=lr_patience
        self.clipgrad=clipgrad

        self.init_lr=args.lr
        self.sbatch=args.sbatch
        self.nIntervals=args.nIntervals

        self.samples=args.samples
        self.lambda_=1.
        self.optimizer= optimizer

        self.modules_names_with_cls = self.find_modules_names(with_classifier=True)
        self.modules_names_without_cls = self.find_modules_names(with_classifier=False)

    def train_Interval(self, dataloader_dict, domain_id = None, sample = True):
        self.model.train()
        w1 = 1.e-3
        w2 = 1.e-3
        w3 = 5.e-2
        for dataname, dataloader in dataloader_dict.items():
            with tqdm(dataloader, total=self.args.num_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.model.zero_grad()
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    if train_inputs.size(2) == 1:
                        train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                   
                    train_embeddings = self.model(train_inputs, domain_id, sample)
                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)

                    test_embeddings = self.model(test_inputs, domain_id, sample)
                    prototypes = get_prototypes(train_embeddings, train_targets, args.num_way)
                    nll = w3*prototypical_loss(prototypes, test_embeddings, test_targets)
                    lp, lv = self.logs()

                    log_var = w1*torch.as_tensor(lv, device=self.args.device).mean()
                    log_p = w2*torch.as_tensor(lp, device=self.args.device).mean()
                    loss = (log_var - log_p)/100 + nll
                    loss.backward(retain_graph=True)           
                    self.optimizer.step()
                    if batch_idx >= args.num_batches:
                        break

    def update_lr(self,t, lr=None, adaptive_lr=False):
        params_dict = []
        if t==0:
            params_dict.append({'params': self.model.parameters(), 'lr': self.init_lr})
        else:
            for name in self.modules_names_without_cls:
                n = name.split('.')
                if len(n) == 1:
                    m = self.model._modules[n[0]]
                elif len(n) == 2:
                    m = self.model._modules[n[0]]._modules[n[1]]
                elif len(n) == 3:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
                elif len(n) == 4:
                    m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
                else:
                    print (name)

                if adaptive_lr is True:
                    params_dict.append({'params': m.weight_rho, 'lr': lr})
                    params_dict.append({'params': m.bias_rho, 'lr': lr})

                else:
                    w_unc = torch.log1p(torch.exp(m.weight_rho.data))
                    b_unc = torch.log1p(torch.exp(m.bias_rho.data))
                    
                    scale = 0.5
                    params_dict.append({'params': m.weight_mu, 'lr': scale*torch.mul(w_unc,self.init_lr)})
                    params_dict.append({'params': m.bias_mu, 'lr': scale*torch.mul(b_unc,self.init_lr)})
                    params_dict.append({'params': m.weight_rho, 'lr':scale*self.init_lr})
                    params_dict.append({'params': m.bias_rho, 'lr':scale*self.init_lr})

        return params_dict

    def find_modules_names(self, with_classifier=False):
        modules_names = []
        for name, p in self.model.named_parameters():
            if with_classifier is False:
                if not name.startswith('classifier'):
                    n = name.split('.')[:-1]
                    modules_names.append('.'.join(n))
            else:
                n = name.split('.')[:-1]
                modules_names.append('.'.join(n))

        modules_names = set(modules_names)
        return modules_names


    def logs(self):

        #print('self.modules_names_without_cls', self.modules_names_without_cls)
        lp, lvp = 0.0, 0.0
        for name in self.modules_names_without_cls:
            #print('before name', name)
            n = name.split('.')
            #print('after name', n, len(n))
            if len(n) == 1:
                m = self.model._modules[n[0]]
            elif len(n) == 2:
                m = self.model._modules[n[0]]._modules[n[1]]
            elif len(n) == 3:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]
            elif len(n) == 4:
                m = self.model._modules[n[0]]._modules[n[1]]._modules[n[2]]._modules[n[3]]
            #print('name mlog', m)
            lp += m.log_prior
            lvp += m.log_variational_posterior

        return lp, lvp


    def save(self, Interval, filepath):
        # Save model
        if self.args.output_folder is not None:
            filename = os.path.join(filepath, 'Interval{0}.pt'.format(Interval))
            with open(filename, 'wb') as f:
                state_dict = self.model.state_dict()
                torch.save(state_dict, f)

    def load(self, Interval):
        str_save = '_'.join(datanames)
        filepath = os.path.join(self.args.output_folder, 'protonet_{}'.format(str_save), 'shot{}'.format(self.args.num_shot), 'way{}'.format(args.num_way))
        filename = os.path.join(filepath, 'Interval{0}.pt'.format(Interval))
        self.model.load_state_dict(torch.load(filename))
        return model

    def valid(self, Interval, dataloader_dict, domain_id, sample=False):
        self.model.eval()
        acc_list = []
        acc_dict = {}
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

                        train_embeddings = self.model(train_inputs, domain_id, sample)
                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings = self.model(test_inputs, domain_id, sample)
                        prototypes = get_prototypes(train_embeddings, train_targets, self.args.num_way)
                        accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                        acc_list.append(accuracy.cpu().data.numpy())
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break

            avg_accuracy = np.round(np.mean(acc_list), 4)
            acc_dict = {dataname:avg_accuracy}

            return acc_dict
        
    def criterion(self,masks):
        reg=0
        count=0
        if self.mask_pre is not None:
            for m,mp in zip(masks,self.mask_pre):
                aux=1-mp
                reg+=(m*aux).sum()
                count+=aux.sum()
        else:
            for m in masks:
                reg+=m.sum()
                count+=np.prod(m.size()).item()
        reg/=count
        return self.lamb*reg

    def train(self, train_loader_list, test_loader_list):

        str_save = '_'.join(datanames)
        filepath = os.path.join(args.output_folder, 'protonet_Bayes_{}'.format(str_save), 'shot{}'.format(args.num_shot), 'way{}'.format(args.num_way))
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        all_accdict = {}


        domain_acc = []
        each_Interval = self.args.num_Interval
        for loaderindex, train_loader in enumerate(train_loader_list):
            params_dict = self.update_lr(loaderindex)
            self.optimizer = BayesianAdam(params=params_dict)
            lr = self.init_lr
            patience = self.lr_patience
            best_acc = 0


            for Interval in range(each_Interval*loaderindex, each_Interval*(loaderindex+1)):
                print('Interval {}'.format(Interval))
                self.train_Interval(train_loader, domain_id = loaderindex)


                total_acc = 0.0
                Interval_acc = []
                for index, test_loader in enumerate(test_loader_list[:loaderindex+1]):
                    test_accuracy_dict = self.valid(Interval, test_loader, domain_id = index)
                    Interval_acc.append(test_accuracy_dict)
                    acc = list(test_accuracy_dict.values())[0]
                    total_acc += acc

                    if Interval == (each_Interval*(loaderindex+1)-1) and index == loaderindex:
                        domain_acc.append(test_accuracy_dict)
                avg_acc = total_acc/(loaderindex+1)
                print('average testing accuracy', avg_acc)

                self.save(Interval, filepath)
                all_accdict[str(Interval)] = Interval_acc
                with open(filepath + '/stats_acc.pickle', 'wb') as handle:
                    pickle.dump(all_accdict, handle, protocol=pickle.HIGHEST_PROTOCOL)

            if loaderindex>0:
                BWT = 0.0            
                for index, (best_domain, Interval_domain) in enumerate(zip(domain_acc, Interval_acc)):
                    best_acc = list(best_domain.values())[0]
                    each_acc = list(Interval_domain.values())[0]
                    BWT += each_acc - best_acc
                avg_BWT = BWT/index
                print('avg_BWT', avg_BWT)
        

def main(args):


    train_loader_list, valid_loader_list, test_loader_list = dataset(args, datanames)
    model = network.Net(args)
    model.to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    appr= PNetUCB(model, optimizer, args=args)
    appr.train(train_loader_list, test_loader_list)


def set_gpu(x):
    x = [str(e) for e in x]
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(x)
    print('using gpu:', ','.join(x))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')

    parser.add_argument('--data_path', type=str, default='',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=40,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=40,
        help='Number of channels for each convolutional layer (default: 64).')
    parser.add_argument('--output_folder', type=str, default='output/datasset/',
        help='Path to the output folder for saving the model (optional).')
    parser.add_argument('--batch-size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks (default: 16).')

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

    parser.add_argument('--num-batches', type=int, default=200,
        help='Number of batches the prototypical network is trained over (default: 100).')
    parser.add_argument('--num_valid_batches', type=int, default=150,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num_memory_batches', type=int, default=1,
        help='Number of batches the model is trained over (default: 150).')
    parser.add_argument('--num-workers', type=int, default=1,
        help='Number of workers for data loading (default: 1).')
    parser.add_argument('--num_query', type=int, default=10,
        help='Number of query examples per class (k in "k-query", default: 15).')
    parser.add_argument('--download', action='store_true',
        help='Download the Omniglot dataset in the data folder.')
    parser.add_argument('--use-cuda', action='store_true',
        help='Use CUDA if available.')
    parser.add_argument('--num_Interval', type=int, default=25,
        help='Number of Intervals for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=5,
        help='Number of tasks in a mini-batch of tasks for validation (default: 4).')
    parser.add_argument('--gpu', type=int, nargs='+', default=[3], help='0 = CPU.')
    parser.add_argument('--seed',               default=0,              type=int,   help='(default=%(default)d)')
    parser.add_argument('--device',             default='cuda:0',       type=str,   help='gpu id')
    # Training parameters
    parser.add_argument('--output',             default='',                     type=str,   help='')
    parser.add_argument('--nIntervals',            default=200,            type=int,   help='')
    parser.add_argument('--sbatch',             default=64,             type=int,   help='')
    parser.add_argument('--lr',                 default=1e-3,           type=float, help='')  # use 0.3 for non-mnist datasets
    parser.add_argument('--nlayers',            default=1,              type=int,   help='')
    parser.add_argument('--nhid',               default=1200,           type=int, help='')

    # UCB HYPER-PARAMETERS
    parser.add_argument('--samples',            default='10',           type=int,     help='Number of Monte Carlo samples')
    parser.add_argument('--rho',                default='-3',           type=float,   help='Initial rho')
    parser.add_argument('--sig1',               default='0.0',          type=float,   help='STD foor the 1st prior pdf in scaled mixture Gaussian')
    parser.add_argument('--sig2',               default='6.0',          type=float,   help='STD foor the 2nd prior pdf in scaled mixture Gaussian')
    parser.add_argument('--pi',                 default='0.25',         type=float,   help='weighting factor for prior')
    parser.add_argument('--taskcla',            default=1,              type=int,     help='number of training domains')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    args.taskcla = len(datanames)

    print('args.device', args.device)
    main(args)

