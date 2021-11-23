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
        self.readout_num = args.readout_num
        
        if args.cifar_svhn:
            self.input_size = 32 * 32 * 3
        else:
            self.input_size = 28 * 28 # MNIST

        self.init_final_layers()

        """Initialize all parameters"""
        # each one stores one layer of modules, i.e. one layer of PathNet
        self.layers = nn.ModuleList()
        for l in range(self.L):
            self.layers.append(nn.ModuleList())
            for m in range(self.M):
                self.layers[-1].append(None)

        self.init(None)

    def init(self, optimal_path):
        """
        If optimal_path is None, then initialize every modules, add a new final layer, 
            backpropogate gradients to every module and the new final layer.

        If optimal_path is not None:
            modules that are not present in optimal_path will be reinitialized, modules 
            present in optimal_path will not be touched. A new final layer will be added. 
            Gradients will only be backpropogated to modules that are not in optimal_path. 


        """
        if optimal_path is None:
            optimal_path = [[None]] * self.L

        neuron_num = self.neuron_num
        module_num = [self.M] * self.L

        # loop through L layers of PathNet
        for l in range(self.L):
            # loop through M modules (each module in PathNet)
            for m in range(module_num[l]):  
                # if this module is not in an optimal path (then its parameters can be reinitialized for transfer learning)
                if m not in optimal_path[l]:
                    if l == 0:
                        curr_module_input_size = self.input_size
                    else:
                        curr_module_input_size = neuron_num
                    # initialization or reinitialization
                    module_ = nn.Linear(curr_module_input_size, neuron_num)
                    self.layers[l][m] = module_

        """final layer which is not inclued in pathnet. Independent for each task"""
        final_layer_ = nn.Linear(neuron_num, self.readout_num)
        self.final_layers.append(final_layer_)

        # select parameters to train (modules not in optimal path + the newest final layer)
        trainable_params = []
        for modules_in_optimal_path_curr_layer, layer_ in zip(optimal_path, self.layers):
            for module_idx, module_ in enumerate(layer_):
                if module_idx in modules_in_optimal_path_curr_layer:
                    # if module is in the optimal path, then do not backpropogate gradients to it
                    module_.requires_grad = False
                else:
                    # if module is not in the optimal path, optimize it using gradient descent
                    trainable_params.append({"params": module_.parameters()})
        trainable_params.append({"params": self.final_layers[-1].parameters()})

        self.optimizer = optim.SGD(trainable_params, lr=self.args.lr)

        if self.args.cuda:
            self.cuda()

    def forward(self, x, path, final_layer_idx):
        x = x.view(-1, self.input_size)

        for l in range(self.L):
            y = F.relu(self.layers[l][path[l][0]](x))
            for j in range(1, self.N):
                y += F.relu(self.layers[l][path[l][j]](x))
            x = y

        x = self.final_layers[final_layer_idx](x)
        return x

    def init_final_layers(self):
        self.final_layers = nn.ModuleList()

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
