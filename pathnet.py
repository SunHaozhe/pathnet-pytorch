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
        self.final_layers = []
        self.init(None)

    def init(self, best_path):
        if best_path is None:
            best_path = [[None]] * self.args.L

        neuron_num = self.args.neuron_num
        module_num = [self.args.M] * self.args.L
        #module_num = self.args.module_num

        """Initialize all parameters"""
        # each one stores one layer of modules, i.e. one layer of PathNet
        self.fc1 = nn.ModuleList()
        self.fc2 = nn.ModuleList()
        self.fc3 = nn.ModuleList()

        # loop through M modules (each module in PathNet)
        for i in range(module_num[0]):  
            # if this module is not in an optimal path (then its parameters can be reinitialized for transfer learning)
            if i not in best_path[0]:  
                """All parameters should be declared as member variable, so I think this is the simplest way to do so"""
                """A better solution: use nn.ModuleList()"""
                if not self.args.cifar_svhn:
                    intput_layer = nn.Linear(28*28, neuron_num)
                else:
                    intput_layer = nn.Linear(32*32*3, neuron_num)
            self.fc1.append(intput_layer)

        for i in range(module_num[1]):
            if i not in best_path[1]:
                layer1 = nn.Linear(neuron_num, neuron_num)
            self.fc2.append(layer1)

        for i in range(module_num[2]):
            if i not in best_path[2]:
                layer2 = nn.Linear(neuron_num, neuron_num)
            self.fc3.append(layer2)

        """final layer which is not inclued in pathnet. Independent for each task"""
        if len(self.final_layers) < 1:
            self.final_layer1 = nn.Linear(neuron_num, self.args.readout_num)
            self.final_layers.append(self.final_layer1)
        else:
            self.final_layer2 = nn.Linear(neuron_num, self.args.readout_num)
            self.final_layers.append(self.final_layer2)

        trainable_params = []
        params_set = [self.fc1, self.fc2, self.fc3]
        for path, params in zip(best_path, params_set):
            for i, param in enumerate(params):
                if i in path:
                    param.requires_grad = False
                else:
                    p = {'params': param.parameters()}
                    trainable_params.append(p)
                    
        p = {'params': self.final_layers[-1].parameters()}
        trainable_params.append(p)
        self.optimizer = optim.SGD(trainable_params, lr=self.args.lr)

        if self.args.cuda:
            self.cuda()

    def forward(self, x, path, last):
        if not self.args.cifar_svhn:
            x = x.view(-1, 28 * 28)
        else:
            x = x.view(-1, 32 * 32 * 3)
        
        # M = self.args.M  # (chongyi zheng): comment out redundant code
        #for i in range(self.args.L):
        y = F.relu(self.fc1[path[0][0]](x))
        for j in range(1, self.args.N):
            y += F.relu(self.fc1[path[0][j]](x))
        x = y

        y = F.relu(self.fc2[path[1][0]](x))
        for j in range(1, self.args.N):
            y += F.relu(self.fc2[path[1][j]](x))
        x = y

        y = F.relu(self.fc3[path[2][0]](x))
        for j in range(1, self.args.N):
            y += F.relu(self.fc3[path[2][j]](x))
        x = y

        '''
        x = F.relu(self.fc1[path[0][0]](x)) + F.relu(self.fc1[path[0][1]](x)) + F.relu(self.fc1[path[0][2]](x))
        x = F.relu(self.fc2[path[1][0]](x)) + F.relu(self.fc2[path[1][1]](x)) + F.relu(self.fc2[path[1][2]](x))
        x = F.relu(self.fc3[path[2][0]](x)) + F.relu(self.fc3[path[2][1]](x)) + F.relu(self.fc3[path[2][2]](x))
        '''
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
