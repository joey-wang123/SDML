import torch
import random
from torchmeta.utils.prototype import get_prototypes, prototypical_loss
import torch.nn.functional as F

def get_accuracy(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)
    _, predictions = torch.min(sq_distances, dim=-1)

    return torch.mean(predictions.eq(targets).float())



def get_accuracy_pred(prototypes, embeddings, targets):
    """Compute the accuracy of the prototypical network on the test/query points.

    Parameters
    ----------
    prototypes : `torch.FloatTensor` instance
        A tensor containing the prototypes for each class. This tensor has shape 
        `(meta_batch_size, num_classes, embedding_size)`.
    embeddings : `torch.FloatTensor` instance
        A tensor containing the embeddings of the query points. This tensor has 
        shape `(meta_batch_size, num_examples, embedding_size)`.
    targets : `torch.LongTensor` instance
        A tensor containing the targets of the query points. This tensor has 
        shape `(meta_batch_size, num_examples)`.

    Returns
    -------
    accuracy : `torch.FloatTensor` instance
        Mean accuracy on the query points.
    """
    sq_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)

    
    prototypes -= prototypes.min(-1, keepdim=True)[0]
    prototypes /= prototypes.max(-1, keepdim=True)[0]

    embeddings -= embeddings.min(-1, keepdim=True)[0]
    embeddings /= embeddings.max(-1, keepdim=True)[0]
    norm_distances = torch.sum((prototypes.unsqueeze(1)
        - embeddings.unsqueeze(2)) ** 2, dim=-1)

    tau = 1.0
    norm_distances = norm_distances/tau
    softprob = -1.0*F.softmax(norm_distances, dim=-1) * F.log_softmax(norm_distances, dim=-1)
    min_dist, predictions = torch.min(sq_distances, dim=-1)
    return torch.mean(predictions.eq(targets).float()), softprob, predictions


def rep_memory_RS(args, model, memory_dict_list):

    memory_loss =0
    for memory_dict in memory_dict_list:
        memory_train_inputs, memory_train_targets = memory_dict['train'] 
        memory_train_inputs = memory_train_inputs.to(device=args.device)[0:1]
        memory_train_targets = memory_train_targets.to(device=args.device)[0:1]
        if memory_train_inputs.size(2) == 1:
            memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
        memory_train_embeddings = model(memory_train_inputs)

        memory_test_inputs, memory_test_targets = memory_dict['test'] 
        memory_test_inputs = memory_test_inputs.to(device=args.device)[0:1]
        memory_test_targets = memory_test_targets.to(device=args.device)[0:1]
        if memory_test_inputs.size(2) == 1:
            memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

        memory_test_embeddings = model(memory_test_inputs)
        memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
        memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

    return memory_loss




def rep_memory(args, model, memory_train):
        memory_loss =0
        for dataidx, dataloader_dict in enumerate(memory_train):
                for dataname, memory_list in dataloader_dict.items():
                    select = random.choice(memory_list)
                    memory_train_inputs, memory_train_targets = select['train'] 
                    memory_train_inputs = memory_train_inputs.to(device=args.device)
                    memory_train_targets = memory_train_targets.to(device=args.device)
                    if memory_train_inputs.size(2) == 1:
                        memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
                    memory_train_embeddings = model(memory_train_inputs, dataidx)

                    memory_test_inputs, memory_test_targets = select['test'] 
                    memory_test_inputs = memory_test_inputs.to(device=args.device)
                    memory_test_targets = memory_test_targets.to(device=args.device)
                    if memory_test_inputs.size(2) == 1:
                        memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

                    memory_test_embeddings = model(memory_test_inputs, dataidx)
                    memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
                    memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)

        return memory_loss





def rep_memory_dict(args, model, memory_train):

        memory_dict = {}
        memory_loss = 0.0

        for dataidx, select in enumerate(memory_train):

            memory_train_inputs, memory_train_targets = select['train'] 
            memory_train_inputs = memory_train_inputs.to(device=args.device)[0:1]
            memory_train_targets = memory_train_targets.to(device=args.device)[0:1]

            if memory_train_inputs.size(2) == 1:
                memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
            memory_train_embeddings = model(memory_train_inputs, dataidx)

            memory_test_inputs, memory_test_targets = select['test'] 
            memory_test_inputs = memory_test_inputs.to(device=args.device)[0:1]
            memory_test_targets = memory_test_targets.to(device=args.device)[0:1]
            if memory_test_inputs.size(2) == 1:
                memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

            memory_test_embeddings = model(memory_test_inputs, dataidx)
            memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
            memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)
                    
        return memory_dict, memory_loss




def rep_memory_agnostic(args, model, memory_train):

        memory_dict = {}
        memory_loss = 0.0

        for dataidx, select in enumerate(memory_train):

            memory_train_inputs, memory_train_targets = select['train'] 
            memory_train_inputs = memory_train_inputs.to(device=args.device)[0:1]
            memory_train_targets = memory_train_targets.to(device=args.device)[0:1]

            if memory_train_inputs.size(2) == 1:
                memory_train_inputs = memory_train_inputs.repeat(1, 1, 3, 1, 1)
            memory_train_embeddings = model(memory_train_inputs, dataidx)[0]

            memory_test_inputs, memory_test_targets = select['test'] 
            memory_test_inputs = memory_test_inputs.to(device=args.device)[0:1]
            memory_test_targets = memory_test_targets.to(device=args.device)[0:1]
            if memory_test_inputs.size(2) == 1:
                memory_test_inputs = memory_test_inputs.repeat(1, 1, 3, 1, 1)

            memory_test_embeddings = model(memory_test_inputs, dataidx)[0]
            memory_prototypes = get_prototypes(memory_train_embeddings, memory_train_targets, args.num_way)
            memory_loss += prototypical_loss(memory_prototypes, memory_test_embeddings, memory_test_targets)
                    
        return memory_dict, memory_loss
