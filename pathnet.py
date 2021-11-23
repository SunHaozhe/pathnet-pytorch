from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.autograd import gradcheck

class Net(nn.Module):

    def __init__(self, args):
        super(Net, self).__init__()
        self.args = args
        self.L = args.L # number of layers
        self.M = args.M # number of modules in each layer
        self.N = args.N # number of active modules
        self.neuron_num = args.neuron_num # number of neuron in each module
        self.final_layers = []
        if args.cifar_svhn:
            self.input_size = 32 * 32 * 3
        else:
            self.input_size = 28 * 28
        self.init(None)

    def init(self, best_path):
        if best_path is None:
            best_path = [[None]] * self.L

        neuron_num = self.neuron_num
        module_num = [self.M] * self.L
        #module_num = self.args.module_num

        """Initialize all parameters"""
        # each one stores one layer of modules, i.e. one layer of PathNet
        self.layers = nn.ModuleList()
        for l in range(self.L):
            self.layers.append(nn.ModuleList())
        
        # loop through L layers of PathNet
        for l in range(self.L):
            # loop through M modules (each module in PathNet)
            for i in range(module_num[l]):  
                # if this module is not in an optimal path (then its parameters can be reinitialized for transfer learning)
                if i not in best_path[l]:  
                    """All parameters should be declared as member variable: use nn.ModuleList()"""
                    if l == 0:
                        curr_module_input_size = self.input_size
                    else:
                        curr_module_input_size = neuron_num
                    # initialization or reinitialization
                    module_ = nn.Linear(curr_module_input_size, neuron_num)
                self.layers[l].append(module_)

        """final layer which is not inclued in pathnet. Independent for each task"""
        if len(self.final_layers) < 1:
            self.final_layer1 = nn.Linear(neuron_num, self.args.readout_num)
            self.final_layers.append(self.final_layer1)
        else:
            self.final_layer2 = nn.Linear(neuron_num, self.args.readout_num)
            self.final_layers.append(self.final_layer2)

        trainable_params = []
        for path, layer_ in zip(best_path, self.layers):
            for i, module_ in enumerate(layer_):
                if i in path:
                    module_.requires_grad = False
                else:
                    p = {'params': module_.parameters()}
                    trainable_params.append(p)
                    
        p = {'params': self.final_layers[-1].parameters()}
        trainable_params.append(p)
        self.optimizer = optim.SGD(trainable_params, lr=self.args.lr)

        if self.args.cuda:
            self.cuda()

    def forward(self, x, path, last):
        x = x.view(-1, self.input_size)

        for l in range(self.L):
            y = F.relu(self.layers[l][path[l][0]](x))
            for j in range(1, self.N):
                y += F.relu(self.layers[l][path[l][j]](x))
            x = y

        x = self.final_layers[last](x)
        return x

    def train_model(self, train_loader, path, num_batch):
        self.train()
        fitness = 0
        train_len = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self.forward(data, path, -1)  # (chongyi zheng): update from '_call_impl' to 'forward' method
            pred = output.max(dim=1)[1]  # get the index of the max log-probability
            fitness += pred.eq(target).cpu().sum()
            train_len += len(target)
            # output: <class 'torch.Tensor'> torch.Size([16, 2]) torch.float32
            # <class 'torch.Tensor'> torch.Size([16]) torch.int64
            loss = F.cross_entropy(output, target)
            loss.backward()
            self.optimizer.step()
            if not batch_idx < num_batch - 1:
                break
        fitness = fitness / train_len
        return fitness

    def test_model(self, test_loader, path, last):
        self.eval()
        fitness = 0
        train_len = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            if self.args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            self.optimizer.zero_grad()
            output = self(data, path, last)
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            fitness += pred.eq(target.data).cpu().sum()
            train_len += len(target.data)
            if batch_idx > 1000:
                break
        fitness = fitness / train_len
        return fitness
