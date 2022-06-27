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
from model import PrototypicalNetworkHAT
from utils import get_accuracy
import numpy as np
from datasets import *
import pickle
import random


datanames = ['Quickdraw', 'Aircraft', 'CUB', 'MiniImagenet', 'Omniglot', 'Plantae', 'Electronic', 'CIFARFS', 'Fungi', 'Necessities']

class PNetHAT(object):
    def __init__(self,model,optimizer, nIntervals=100, clipgrad=10000,lamb=0.75,smax=400,args=None):
        self.args = args
        self.model=model
        self.nIntervals=nIntervals
        self.clipgrad=clipgrad
        self.ce=torch.nn.CrossEntropyLoss()
        self.optimizer= optimizer

        self.lamb=lamb  
        self.smax=smax      
        self.mask_pre=None
        self.mask_back=None

        str_save = '_'.join(datanames)
        self.filepath = os.path.join(self.args.output_folder, 'protonet_HAT_{}'.format(str_save), 'shot{}'.format(self.args.num_shot), 'way{}'.format(args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)


    def train_Interval(self, Interval, dataloader_dict, domain_id = None):
        self.model.train()
        r = self.args.num_batches*self.args.batch_size
        for dataname, dataloader in dataloader_dict.items():
            with tqdm(dataloader, total=self.args.num_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.model.zero_grad()
                    i = batch_idx*self.args.batch_size
                    s=(self.smax-1/self.smax)*i/r+1/self.smax
                    train_inputs, train_targets = batch['train']
                    train_inputs = train_inputs.to(device=self.args.device)
                    train_targets = train_targets.to(device=self.args.device)
                    if train_inputs.size(2) == 1:
                        train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                    train_embeddings, masks = self.model(train_inputs, domain_id, s=s)

                    test_inputs, test_targets = batch['test']
                    test_inputs = test_inputs.to(device=self.args.device)
                    test_targets = test_targets.to(device=self.args.device)
                    if test_inputs.size(2) == 1:
                        test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                    test_embeddings, masks = self.model(test_inputs, domain_id, s=s)
                    prototypes = get_prototypes(train_embeddings, train_targets, args.num_way)
                    loss = prototypical_loss(prototypes, test_embeddings, test_targets)
                    mask_loss = self.criterion(masks)
                    loss+=mask_loss                      
                    loss.backward()

                    thres_cosh=50
                    thres_emb=6
                    
                    if domain_id>0:
                        for n,p in self.model.named_parameters():
                            if n in self.mask_back:
                                p.grad.data*=self.mask_back[n]

                    for n,p in self.model.named_parameters():
                        if n.startswith('e'):
                            num=torch.cosh(torch.clamp(s*p.data,-thres_cosh,thres_cosh))+1
                            den=torch.cosh(p.data)+1
                            p.grad.data*=self.smax/s*num/den

                    # Apply step
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.clipgrad)
                    
                    self.optimizer.step()
                    for n,p in self.model.named_parameters():
                        if n.startswith('e'):
                            p.data=torch.clamp(p.data,-thres_emb,thres_emb)
                    

                    if batch_idx >= args.num_batches:
                        break
                
    def save(self, Interval):
        if self.args.output_folder is not None:
            filename = os.path.join(self.filepath, 'Interval{0}.pt'.format(Interval))
            with open(filename, 'wb') as f:
                state_dict = self.model.state_dict()
                torch.save(state_dict, f)

    def load(self, Interval):

        args.output_folder = 'output/datasset/'
        str_save = '_'.join(datanames)
        filepath = os.path.join(self.args.output_folder, 'protonet_{}'.format(str_save), 'shot{}'.format(self.args.num_shot), 'way{}'.format(args.num_way))
        filename = os.path.join(filepath, 'Interval{0}.pt'.format(Interval))
        self.model.load_state_dict(torch.load(filename))
        return model

    def valid(self, Interval, dataloader_dict, domain_id):
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

                        train_embeddings, masks = self.model(train_inputs, domain_id, s=self.smax)
                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings, masks = self.model(test_inputs, domain_id, s=self.smax)
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
        each_Interval = self.args.num_Interval
        all_accdict = {}


        domain_acc = []
        for loaderindex, train_loader in enumerate(train_loader_list):
            for Interval in range(each_Interval*loaderindex, each_Interval*(loaderindex+1)):
                print('Interval {}'.format(Interval))
                self.train_Interval(Interval, train_loader, domain_id = loaderindex)


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

                self.save(Interval)
                all_accdict[str(Interval)] = Interval_acc
                with open(self.filepath + '/stats_acc.pickle', 'wb') as handle:
                    pickle.dump(all_accdict, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            mask=self.model.mask(device = self.args.device, t =loaderindex , s=self.smax)
            for i in range(len(mask)):
                mask[i]=torch.autograd.Variable(mask[i].data.clone(),requires_grad=False)
            if loaderindex==0:
                self.mask_pre=mask
            else:
                for i in range(len(self.mask_pre)):
                    self.mask_pre[i]=torch.max(self.mask_pre[i],mask[i])

            # Weights mask
            self.mask_back={}
            for n,_ in self.model.named_parameters():
                vals=self.model.get_view_for(n,self.mask_pre)
                if vals is not None:
                    self.mask_back[n]=1-vals


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
    model = PrototypicalNetworkHAT(3,
                                args.embedding_size,
                                hidden_size=args.hidden_size, num_tasks=len(datanames))
    model.to(device=args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    seqmeta= PNetHAT(model, optimizer, args=args)
    seqmeta.train(train_loader_list, test_loader_list)
        



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Prototypical Networks')

    parser.add_argument('--data_path', type=str, default='/data/',
        help='Path to the folder the data is downloaded to.')
    parser.add_argument('--num-shot', type=int, default=5,
        help='Number of examples per class (k in "k-shot", default: 5).')
    parser.add_argument('--num-way', type=int, default=5,
        help='Number of classes per task (N in "N-way", default: 5).')

    parser.add_argument('--embedding-size', type=int, default=64,
        help='Dimension of the embedding/latent space (default: 64).')
    parser.add_argument('--hidden-size', type=int, default=64,
        help='Number of channels for each convolutional layer (default: 64).')

    parser.add_argument('--output_folder', type=str, default='output/newsavedir/',
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
    parser.add_argument('--VGGflower_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for VGGflower (default: 4).')
    parser.add_argument('--Fungi_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Fungiflower (default: 4).')
    parser.add_argument('--Quickdraw_batch_size', type=int, default=2,
        help='Number of tasks in a mini-batch of tasks for Quickdraw (default: 4).')
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
    parser.add_argument('--num_Interval', type=int, default=20,
        help='Number of Intervals for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for validation (default: 4).')
    parser.add_argument('--gpu', type=int, nargs='+', default=[0], help='0 = CPU.')

    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('args.device', args.device)
    main(args)
