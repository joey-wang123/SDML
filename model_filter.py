import torch.nn as nn
import torch

def conv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

def conv3x3nopool(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )


def conv3x3nobatch(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )



def conv3x3_2(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, **kwargs),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )


class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0, num_block = 1):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks
        self.num_block = num_block
        block_size = int(hidden_size/num_block)

        self.conv1 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv1.append(conv3x3nobatch(in_channels, block_size))

        self.conv2 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv2.append(conv3x3nobatch(hidden_size, block_size))
  
        self.conv3 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv3.append(conv3x3nobatch(hidden_size, block_size))

        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = nn.Sequential(
                conv3x3(hidden_size, hidden_size),
                conv3x3(hidden_size, out_channels)
            )
            self.domain_out.append(self.task)
        
        
    def forward(self, inputs, domain_id, s=1):
        catlayer1 = []
        for ind in range(self.num_block):
             catlayer1.append(self.conv1[ind](inputs.view(-1, *inputs.shape[2:])))
        h = torch.cat(catlayer1, 1)

        catlayer2 = []
        for ind in range(self.num_block):
             catlayer2.append(self.conv2[ind](h))
        h = torch.cat(catlayer2, 1)

        catlayer3 = []
        for ind in range(self.num_block):
             catlayer3.append(self.conv3[ind](h))
        h = torch.cat(catlayer3, 1)

        h = self.domain_out[domain_id](h)
        return h.view(*inputs.shape[:2], -1)

  
    def set_req_grad(self, domain_id, req_grad):

        for i in range(self.taskcla):
            if i!= domain_id:
                params = list(self.domain_out[i].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = req_grad
            else:
                params = list(self.domain_out[domain_id].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = True
        return






class PrototypicalNetworkhead1(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0, num_block = 1):
        super(PrototypicalNetworkhead1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks
        self.num_block = num_block
        block_size = int(hidden_size/num_block)


        self.conv1 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv1.append(conv3x3nobatch(in_channels, block_size))

        self.conv2 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv2.append(conv3x3nobatch(hidden_size, block_size))
  
        self.conv3 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv3.append(conv3x3nobatch(hidden_size, block_size))

        self.conv4 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv4.append(conv3x3nobatch(hidden_size, block_size))

        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = nn.Sequential(
                conv3x3(hidden_size, out_channels)
            )
            self.domain_out.append(self.task)
        
        
    def forward(self, inputs, domain_id, s=1):

        catlayer1 = []
        for ind in range(self.num_block):
             catlayer1.append(self.conv1[ind](inputs.view(-1, *inputs.shape[2:])))
        h = torch.cat(catlayer1, 1)

        catlayer2 = []
        for ind in range(self.num_block):
             catlayer2.append(self.conv2[ind](h))
        h = torch.cat(catlayer2, 1)

        catlayer3 = []
        for ind in range(self.num_block):
             catlayer3.append(self.conv3[ind](h))
        h = torch.cat(catlayer3, 1)

        catlayer4 = []
        for ind in range(self.num_block):
             catlayer4.append(self.conv4[ind](h))
        h = torch.cat(catlayer4, 1)

        h = self.domain_out[domain_id](h)
        return h.view(*inputs.shape[:2], -1)

  
    def set_req_grad(self, domain_id, req_grad):

        for i in range(self.taskcla):
            if i!= domain_id:
                params = list(self.domain_out[i].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = req_grad
            else:
                params = list(self.domain_out[domain_id].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = True




class PrototypicalNetworkinfer(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0, num_block = 1):
        super(PrototypicalNetworkinfer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks
        self.num_block = num_block
        block_size = int(hidden_size/num_block)
        self.conv1 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv1.append(conv3x3_2(in_channels, block_size))

        self.conv2 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv2.append(conv3x3_2(hidden_size, block_size))
  
        self.conv3 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv3.append(conv3x3_2(hidden_size, block_size))

        self.conv4 = torch.nn.ModuleList()
        for _ in range(self.num_block):
            self.conv4.append(conv3x3_2(hidden_size, block_size))

        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = nn.Sequential(
                conv3x3(hidden_size, out_channels)
            )
            self.domain_out.append(self.task)
        print('self.taskcla', self.taskcla)
        
    def forward(self, inputs, domain_id, test= False):

        catlayer1 = []
        for ind in range(self.num_block):
             catlayer1.append(self.conv1[ind](inputs.view(-1, *inputs.shape[2:])))
        h = torch.cat(catlayer1, 1)

        catlayer2 = []
        for ind in range(self.num_block):
             catlayer2.append(self.conv2[ind](h))
        h = torch.cat(catlayer2, 1)

        catlayer3 = []
        for ind in range(self.num_block):
             catlayer3.append(self.conv3[ind](h))
        h = torch.cat(catlayer3, 1)

        catlayer4 = []
        for ind in range(self.num_block):
             catlayer4.append(self.conv4[ind](h))
        h = torch.cat(catlayer4, 1)


        if test:
            out_list = []
            for id in range(domain_id+1):
                htemp = self.domain_out[id](h)
                out_list.append(htemp.view(*inputs.shape[:2], -1))
        else:
            h = self.domain_out[domain_id](h)
            out_list = [h.view(*inputs.shape[:2], -1)]
        return out_list

  
    def set_req_grad(self, domain_id, req_grad):

        for i in range(self.taskcla):
            if i!= domain_id:
                params = list(self.domain_out[i].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = req_grad
            else:
                params = list(self.domain_out[domain_id].parameters()) 
                for ind in range(len(params)):
                    params[ind].requires_grad = True
