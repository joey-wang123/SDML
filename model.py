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


class PrototypicalNetworkHAT(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0):
        super(PrototypicalNetworkHAT, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks

        self.ec1=torch.nn.Embedding(self.taskcla,64)
        self.ec2=torch.nn.Embedding(self.taskcla,64)
        self.ec3=torch.nn.Embedding(self.taskcla,64)
        self.ec4=torch.nn.Embedding(self.taskcla,64)
        
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        self.gate=torch.nn.Sigmoid()

        self.conv1 = conv3x3(in_channels, hidden_size)
        self.conv2 = conv3x3(hidden_size, hidden_size)
        self.conv3 = conv3x3(hidden_size, hidden_size)
        self.conv4 = conv3x3(hidden_size, out_channels)
        

        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = nn.Sequential(
                conv3x3(hidden_size, out_channels)
            )
            self.domain_out.append(self.task)
        
    def forward(self, inputs, domain_id, s=1):

        masks=self.mask(inputs.get_device(), domain_id, s=s)
        gc1,gc2,gc3,gc4 = masks
        h = self.conv1(inputs.view(-1, *inputs.shape[2:]))
        h=h*gc1.view(1,-1,1,1).expand_as(h)
        h = self.conv2(h)
        h=h*gc2.view(1,-1,1,1).expand_as(h)
        h = self.conv3(h)
        h=h*gc3.view(1,-1,1,1).expand_as(h)
        h = self.conv4(h)
        h=h*gc4.view(1,-1,1,1).expand_as(h)

        h = self.domain_out[domain_id](h)
        
        return h.view(*inputs.shape[:2], -1), masks

    def mask(self, device, t, s=1):
        t = torch.tensor(t).to(device)
        gc1=self.gate(s*self.ec1(t))
        gc2=self.gate(s*self.ec2(t))
        gc3=self.gate(s*self.ec3(t))
        gc4=self.gate(s*self.ec4(t))
        return [gc1,gc2,gc3,gc4]

    def get_view_for(self,n,masks):
        gc1,gc2,gc3,gc4 = masks
        if n=='conv1.0.weight':
            return gc1.data.view(-1,1,1,1).expand_as(self.conv1[0].weight)
        elif n=='conv1.0.bias':
            return gc1.data.view(-1)
        elif n=='conv2.0.weight':
            post=gc2.data.view(-1,1,1,1).expand_as(self.conv2[0].weight)
            pre=gc1.data.view(1,-1,1,1).expand_as(self.conv2[0].weight)
            return torch.min(post,pre)
        elif n=='conv2.0.bias':
            return gc2.data.view(-1)
        elif n=='conv3.0.weight':
            post=gc3.data.view(-1,1,1,1).expand_as(self.conv3[0].weight)
            pre=gc2.data.view(1,-1,1,1).expand_as(self.conv3[0].weight)
            return torch.min(post,pre)
        elif n=='conv3.0.bias':
            return gc3.data.view(-1)
        elif n=='conv4.0.weight':
            post=gc4.data.view(-1,1,1,1).expand_as(self.conv4[0].weight)
            pre=gc3.data.view(1,-1,1,1).expand_as(self.conv4[0].weight)
            return torch.min(post,pre)
        elif n=='conv4.0.bias':
            return gc4.data.view(-1)
        return None


class PrototypicalNetwork(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0):
        super(PrototypicalNetwork, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks

        self.conv1 = conv3x3nobatch(in_channels, hidden_size)
        self.conv2 = conv3x3nobatch(hidden_size, hidden_size)
        self.conv3 = conv3x3nobatch(hidden_size, hidden_size)
        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = nn.Sequential(
                conv3x3(hidden_size, hidden_size),
                conv3x3(hidden_size, out_channels)
            )
            self.domain_out.append(self.task)
        
        
    def forward(self, inputs, domain_id, s=1):
        h = self.conv1(inputs.view(-1, *inputs.shape[2:]))
        h = self.conv2(h)
        h = self.conv3(h)
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


class PrototypicalNetworkJoint(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0):
        super(PrototypicalNetworkJoint, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks

        
        self.conv1 = conv3x3(in_channels, hidden_size)
        self.conv2 = conv3x3(hidden_size, hidden_size)
        self.conv3 = conv3x3(hidden_size, hidden_size)
        self.conv4 = conv3x3(hidden_size, hidden_size)
        self.conv5 = conv3x3(hidden_size, out_channels)
        
    def forward(self, inputs, domain_id = None):
        h = self.conv1(inputs.view(-1, *inputs.shape[2:]))
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.conv4(h)
        h = self.conv5(h)
        return h.view(*inputs.shape[:2], -1)




class PrototypicalNetworkMultitask(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64, num_tasks = 0):
        super(PrototypicalNetworkMultitask, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.taskcla = num_tasks

        self.conv1 = conv3x3(in_channels, hidden_size)
        self.conv2 = conv3x3(hidden_size, hidden_size)
        self.conv3 = conv3x3(hidden_size, hidden_size)
        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = nn.Sequential(
                conv3x3(hidden_size, hidden_size),
                conv3x3(hidden_size, out_channels)
            )
            self.domain_out.append(self.task)
        
        
    def forward(self, inputs, domain_id, s=1):
        h = self.conv1(inputs.view(-1, *inputs.shape[2:]))
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.domain_out[domain_id](h)
        return h.view(*inputs.shape[:2], -1)


