import os
import torch
from tqdm import tqdm
import logging
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
from torchvision.transforms import ToTensor, Resize, Compose
from model import PrototypicalNetworkJoint
from utils import get_accuracy
import numpy as np
from datasets import *
import pickle
import random
import time
from copy import deepcopy


datanames = ['Quickdraw', 'Aircraft', 'CUB', 'MiniImagenet', 'Omniglot', 'Plantae', 'Electronic', 'CIFARFS', 'Fungi', 'Necessities']



class PNetMER(object):

    def __init__(self,model, args=None):
        self.args = args
        self.model=model
        self.memory_rep = []
        self.step = 0
        self.str_save = '_'.join(datanames)
        self.filepath = os.path.join(self.args.output_folder, 'protonet_MER{}'.format(self.str_save), 'shot{}'.format(self.args.num_shot), 'way{}'.format(self.args.num_way))
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)

    def train(self,optimizer, dataloader_dict, domain_id = None):

        gamma = 1.0
        num_steps = 5
        beta = 0.03
        for dataname, dataloader in dataloader_dict.items():
            with tqdm(dataloader, total=args.num_batches) as pbar:
                for batch_idx, batch in enumerate(pbar):
                    self.model.zero_grad()
                    before = deepcopy(self.model.state_dict())

                    #Reservoir sampling
                    if self.step < self.args.memory_limit:
                        savedict = batch
                        self.memory_rep.append(savedict)
                    else:
                        randind = random.randint(0, self.step)
                        if randind < self.args.memory_limit:
                            savedict =  batch
                            self.memory_rep[randind] = savedict
                    self.step += 1


                    for step in range(0, num_steps): 
                        weights_before = deepcopy(self.model.state_dict())
                        train_inputs, train_targets = batch['train']
                        train_inputs = train_inputs.to(device=self.args.device)
                        train_targets = train_targets.to(device=self.args.device)
                        if train_inputs.size(2) == 1:
                            train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)

                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=self.args.device)
                        test_targets = test_targets.to(device=self.args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)

                        for (train_input, train_target, test_input, test_target) in zip(train_inputs, train_targets, test_inputs, test_targets):
                            self.model.zero_grad()
                            train_embedding = self.model(train_input.unsqueeze(0), domain_id)
                            test_embedding = self.model(test_input.unsqueeze(0), domain_id)
                            prototypes = get_prototypes(train_embedding, train_target.unsqueeze(0), args.num_way)
                            loss = prototypical_loss(prototypes, test_embedding, test_target.unsqueeze(0))
                            loss.backward()
                            optimizer.step()

                        if self.memory_rep:
                                    select = random.choice(self.memory_rep)
                                    memory_train_inputs, memory_train_targets = select['train'] 
                                    memory_train_inputs = memory_train_inputs.to(device=self.args.device)
                                    memory_train_targets = memory_train_targets.to(device=self.args.device)
                                    if memory_train_inputs.size(2) == 1:
                                        memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)

                                    memory_test_inputs, memory_test_targets = select['test'] 
                                    memory_test_inputs = memory_test_inputs.to(device=self.args.device)
                                    memory_test_targets = memory_test_targets.to(device=self.args.device)
                                    if memory_test_inputs.size(2) == 1:
                                        memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

                                    index = -1
                                    for (memory_train_input, memory_train_target, memory_test_input, memory_test_target) in zip(memory_train_inputs, memory_train_targets, memory_test_inputs, memory_test_targets):
                                        index += 1
                                        if index ==1:
                                            break
                                        self.model.zero_grad()
                                        memory_train_embedding = self.model(memory_train_input.unsqueeze(0))
                                        memory_test_embedding = self.model(memory_test_input.unsqueeze(0))
                                        memory_prototypes = get_prototypes(memory_train_embedding, memory_train_target.unsqueeze(0), args.num_way)
                                        memory_loss = prototypical_loss(memory_prototypes, memory_test_embedding, memory_test_target.unsqueeze(0))

                                        memory_loss.backward()
                                        optimizer.step()

                        weights_after = self.model.state_dict()
                        self.model.load_state_dict({name : weights_before[name] + ((weights_after[name] - weights_before[name]) * beta) for name in weights_before})
                    after = self.model.state_dict()
                    self.model.load_state_dict({name : before[name] + ((after[name] - before[name]) * gamma) for name in before})

                    if batch_idx >= args.num_batches:
                        break

    def save(self, Interval):
        # Save model
        if self.args.output_folder is not None:

            filename = os.path.join(self.filepath, 'Interval{0}.pt'.format(Interval))
            with open(filename, 'wb') as f:
                state_dict = self.model.state_dict()
                torch.save(state_dict, f)

    def load(self, Interval, model):

        self.args.output_folder = 'output/datasset/'
        str_save = '_'.join(datanames)
        filepath = os.path.join(self.args.output_folder, 'protonet_{}'.format(str_save), 'shot{}'.format(args.num_shot), 'way{}'.format(args.num_way))
        filename = os.path.join(filepath, 'Interval{0}.pt'.format(Interval))
        self.model.load_state_dict(torch.load(filename))


    def valid(self, Interval, dataloader_dict, domain_id):
        acc_list = []
        acc_dict = {}
        for dataname, dataloader in dataloader_dict.items():
            with torch.no_grad():
                with tqdm(dataloader, total=self.args.num_valid_batches) as pbar:
                    for batch_idx, batch in enumerate(pbar):
                        self.model.zero_grad()

                        train_inputs, train_targets = batch['train']
                        train_inputs = train_inputs.to(device=args.device)
                        train_targets = train_targets.to(device=args.device)
                        if train_inputs.size(2) == 1:
                            train_inputs = train_inputs.repeat(1, 1, 3, 1, 1)
                        train_embeddings = self.model(train_inputs, domain_id)

                        test_inputs, test_targets = batch['test']
                        test_inputs = test_inputs.to(device=args.device)
                        test_targets = test_targets.to(device=args.device)
                        if test_inputs.size(2) == 1:
                            test_inputs = test_inputs.repeat(1, 1, 3, 1, 1)
                        test_embeddings = self.model(test_inputs, domain_id)
                        prototypes = get_prototypes(train_embeddings, train_targets,
                        self.args.num_way)
                        accuracy = get_accuracy(prototypes, test_embeddings, test_targets)
                        acc_list.append(accuracy.cpu().data.numpy())
                        pbar.set_description('dataname {} accuracy ={:.4f}'.format(dataname, np.mean(acc_list)))
                        if batch_idx >= self.args.num_valid_batches:
                            break

            avg_accuracy = np.round(np.mean(acc_list), 4)
            acc_dict = {dataname:avg_accuracy}
            logging.debug('Interval_{}_{}_accuracy_{}'.format(Interval, dataname, avg_accuracy))
            return acc_dict
        

def main(args):

    all_accdict = {}
    train_loader_list, valid_loader_list, test_loader_list = dataset(args, datanames)
    model = PrototypicalNetworkJoint(3,
                                args.embedding_size,
                                hidden_size=args.hidden_size)
    model.to(device=args.device)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    each_Interval = args.num_Interval
    savemode = 'PNet-MER'
    str_save = '_'.join(datanames)


    seqmeta = PNetMER(model, args=args)
    dataname = []
    
    domain_acc = []
    for loaderindex, train_loader in enumerate(train_loader_list):
        for Interval in range(each_Interval*loaderindex, each_Interval*(loaderindex+1)):
            print('Interval {}'.format(Interval))
            dataname.append(list(train_loader.keys())[0])
            seqmeta.train(optimizer, train_loader, domain_id = loaderindex)

            
            total_acc = 0.0
            Interval_acc = []
            for index, test_loader in enumerate(test_loader_list[:loaderindex+1]):
                test_accuracy_dict = seqmeta.valid(Interval, test_loader, domain_id = index)
                Interval_acc.append(test_accuracy_dict)
                acc = list(test_accuracy_dict.values())[0]
                total_acc += acc

                if Interval == (each_Interval*(loaderindex+1)-1) and index == loaderindex:
                    domain_acc.append(test_accuracy_dict)
            avg_acc = total_acc/(loaderindex+1)
            print('average testing accuracy', avg_acc)
        
            seqmeta.save(Interval)
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
    parser.add_argument('--num_Interval', type=int, default=25,
        help='Number of Intervals for meta train.') 
    parser.add_argument('--valid_batch_size', type=int, default=3,
        help='Number of tasks in a mini-batch of tasks for validation (default: 4).')
    parser.add_argument('--memory_limit', type=int, default=10,
        help='Number of memory tasks.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')

    args = parser.parse_args()
    device = 'cuda:{}'.format(args.gpu) if torch.cuda.is_available() else 'cpu'
    args.device = torch.device(device)

    print('args.device', args.device)
    main(args)

