import torch
import torch.nn as nn
import torch.nn.functional as F

from .BayesianConvs import BayesianConv2D
from .BatchNorm import BayesianBatchNorm2d
from .FC import BayesianLinear

def conv3x3(in_planes, out_planes, args, stride=1):
    return BayesianConv2D(in_planes, out_planes, 3, args=args, stride=stride, padding=1, use_bias=True)

def bayesconv3x3(in_channels, out_channels, **kwargs):
    return nn.Sequential(
        BayesianConv2D(in_channels, out_channels, 1, self.args,
                               stride=stride, use_bias=True),
        BayesianBatchNorm2d(out_channels, self.args),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class BasicBlock(nn.Module):

    def __init__(self, in_channels, out_channels, args, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = BayesianConv2D(in_channels, out_channels, 3, args, stride=1, padding=1,
                                    use_bias=True)
        self.bn1 = BayesianBatchNorm2d(out_channels, args)

    def forward(self, x, sample=False):

        x = self.conv1(x, sample)
        x = self.bn1(x, sample)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, 2)

        return  x

class BayesianNet(nn.Module):

    def __init__(self, args):
 
        super(BayesianNet, self).__init__()

        self.args = args
        self.sig1 = args.sig1
        self.sig2 = args.sig2
        self.pi = args.pi
        self.rho = args.rho

        in_channels = 3
        hidden_size = args.hidden_size
        out_channels = args.embedding_size
        self.taskcla = args.taskcla
        self.block1 = BasicBlock(in_channels, hidden_size, args)
        self.block2 = BasicBlock(hidden_size, hidden_size, args)
        self.block3 = BasicBlock(hidden_size, hidden_size, args)
        self.block4 = BasicBlock(hidden_size, out_channels, args)

        self.domain_out = torch.nn.ModuleList()
        for _ in range(self.taskcla):
            self.task = BasicBlock(hidden_size, hidden_size, args)
            self.domain_out.append(self.task)

    def prune(self, mask_modules):
        for module, mask in mask_modules.items():
            module.prune_module(mask)

    def forward(self, inputs, domain_id, sample=False):

        x = inputs.view(-1, *inputs.shape[2:])
        x = self.block1(x, sample)
        x = self.block2(x, sample)
        x = self.block3(x, sample)
        x = self.block4(x, sample)
        x = self.domain_out[domain_id].forward(x, sample)

        return x.view(*inputs.shape[:2], -1)


def Net(args):
    return BayesianNet(args)