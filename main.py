import os
import argparse
import random
import pickle

import torch
from torchvision import datasets, transforms

import pathnet
import genotype
import mnist_dataset
import svhn_dataset
import cifar_dataset
import visualize
import get_svhn_data

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')


parser.add_argument('--L', type=int, default=3, metavar='N',
                    help='layers')
parser.add_argument('--M', type=int, default=10, metavar='N',
                    help='units in each layer')
parser.add_argument('--N', type=int, default=3, metavar='N',
                    help='number of active units')
parser.add_argument('--pop', type=int, default=64, metavar='N',
                    help='number of gene')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    help='input batch size for training (default: 16)')
parser.add_argument('--num-batch', type=int, default=50, metavar='N',
                    help='input batch number for each episode (default: 50)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--neuron-num', type=int, default=20, metavar='N',
                    help='number of neuron in each module')
parser.add_argument('--generation-limit', type=int, default=100, metavar='N',
                    help='number of generation to compute')
parser.add_argument('--noise-prob', type=float, default=0.5, metavar='N',
                    help='salt and pepper noise rate')
parser.add_argument('--threshold', type=float, default=0.998, metavar='N',
                    help='accuracy threshold to finish the first task')
parser.add_argument('--readout-num', type=int, default=2, metavar='N',
                    help='number of units for readout (default: 2 for MNIST binary classification task)')
parser.add_argument('--control', action='store_true', default=False,
                    help='controlled experiment on/off')
parser.add_argument('--fine-tune', action='store_true', default=False,
                    help='fine-tuning control experiment on/off')
parser.add_argument('--no-graph', dest='vis', action='store_false', default=True,
                    help='show graph')
parser.add_argument('--no-save', action='store_true', default=False,
                    help='do not save result')
parser.add_argument('--cifar-svhn', action='store_true', default=False,
                    help='cifar-svhn task')
parser.add_argument('--trainset-limit', type=int, default=20000, metavar='N',
                    help='training dataset limitation for RAM')
parser.add_argument('--testset-limit', type=int, default=1000, metavar='N',
                    help='test dataset limitation for RAM')
parser.add_argument('--cifar-first', action='store_true', default=False,
                    help='cifar trained first')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def cifar_svhn_data(cifar):
    if cifar:
        print("Extracting cifar dataset...")
        cifar = cifar_dataset.Dataset(args)
        train_loader, test_loader = cifar.return_dataset()
    else:
        print("Extracting cSVHN dataset...")
        svhn = svhn_dataset.Dataset(args)
        train_loader, test_loader = svhn.return_dataset()
    return train_loader, test_loader


def train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, gen, vis_color):
    pathways = gene.sample()
    fitnesses = []
    train_data = [(data, target) for (data, target) in train_loader]
    for pathway in pathways:
        path = pathway.return_genotype()
        fitness = model.train_model(train_data, path, args.num_batch)
        fitnesses.append(fitness)
    print("Generation {} : Fitnesses = {} vs {}".format(gen, fitnesses[0], fitnesses[1]))

    gene.overwrite(pathways, fitnesses)
    genes = gene.return_all_genotypes()
    visualizer.show(genes, vis_color)
    if max(fitnesses) > best_fitness:
        best_fitness = max(fitnesses)
        best_path = pathways[fitnesses.index(max(fitnesses))].return_genotype()
    return best_fitness, best_path, max(fitnesses)


def train_control(model, gene, visualizer, train_loader, gen):        
    path = gene.return_control_genotype()
    train_data = [(data, target) for (data,target) in train_loader]
    fitness = model.train_model(train_data, path, args.num_batch)
    print("Generation {} : Fitness = {}".format(gen, fitness))
    genes = [gene.return_control_genotype()] * args.pop
    visualizer.show(genes, 'm')
    return fitness

def main():
    model = pathnet.Net(args)
    gene = genotype.Genetic(args.L, args.M, args.N, args.pop)
    module_num = [args.M] * args.L
    visualizer = visualize.GraphVisualize(module_num, args.vis)
    
    if args.cuda:
        model.cuda()
    
    if not os.path.isdir('./result'):
        os.makedirs("./result")
    if not os.path.isdir('./data'):
        os.makedirs("./data")

    if not args.cifar_svhn:
        if not os.path.isdir('./data/mnist'):
            os.system('./get_mnist_data.sh')

        if os.path.exists('./result/result_mnist.pickle'):
            f = open('./result/result_mnist.pickle', 'rb')
            result = pickle.load(f)
            f.close()
        else:
            result = []

        prob = args.noise_prob
        dataset = mnist_dataset.Dataset(prob)
        labels = random.sample(range(10), 2)
        print("Two training classes : {} and {}".format(labels[0], labels[1]))
        dataset.set_binary_class(labels[0], labels[1])
        train_loader = dataset.convert2tensor(args)

    else:
        get_svhn_data.download()

        if not os.path.isdir('./data/cifar'):
            os.system('./get_cifar10_data.sh')
        
        if os.path.exists('./result/result_cifar_svhn.pickle'):
            f = open('./result/result_cifar_svhn.pickle', 'rb')
            result = pickle.load(f)
            f.close()
        else:
            result = []

        train_loader, test_loader = cifar_svhn_data(args.cifar_first)


    """first task"""
    print("First task started...")
    best_fitness = 0.0
    best_path = [[None] * args.N] * args.L
    gen = 0
    first_fitness = []
    for gen in range(args.generation_limit):
        if not args.control:
            best_fitness, best_path, max_fitness = train_pathnet(
                model, gene, visualizer, train_loader, best_fitness, best_path, gen, 'm')
            first_fitness.append(max_fitness)

        else:  # control experiment
            fitness = train_control(model, gene, visualizer, train_loader, gen)
            first_fitness.append(fitness)

    print("First task done!! Move to next task")
    print("Second task started...")

    if not args.control:
        gene = genotype.Genetic(args.L, args.M, args.N, args.pop)
        '''
        if not args.cifar_svhn:
            gene = genotype.Genetic(3, 10, 3, 64)
        else:
            gene = genotype.Genetic(3, 20, 5, 64)

        gene = genotype.Genetic(3, 10, 3, 64)
        '''
        model.init(best_path)
        visualizer.set_fixed(best_path, 'r')
    else:
        if not args.fine_tune:
            model = pathnet.Net(args)
            gene = genotype.Genetic(args.L, args.M, args.N, args.pop)
            '''
            if not args.cifar_svhn:
                gene = genotype.Genetic(3, 10, 3, 64)
            else:
                gene = genotype.Genetic(3, 20, 5, 64)
            '''
    #labels = random.sample(range(10), 2)
    if not args.cifar_svhn:
        c_1 = labels[0]
        
        while True:
            c_2 = random.randint(0, 10-1)
            if not c_2 == c_1:
                break
        labels = [c_1, c_2]
        print("Two training classes : {} and {}".format(labels[0], labels[1]))
        dataset.set_binary_class(labels[0], labels[1])
        train_loader = dataset.convert2tensor(args)

    else:
        train_loader, test_loader = cifar_svhn_data(not args.cifar_first)


    best_fitness = 0.0    
    #best_path = [[None] * 3] * 3
    best_path = [[None] * args.N] * args.L
    gen = 0

    second_fitness = []
    for gen in range(args.generation_limit):
        if not args.control:
            best_fitness, best_path, max_fitness = train_pathnet(model, gene, visualizer, train_loader, best_fitness, best_path, gen, 'c')
            second_fitness.append(max_fitness)

        else: ##control experiment
            fitness = train_control(model, gene, visualizer, train_loader, gen)
            second_fitness.append(fitness)

    print("Second task done!! Goodbye!!")

    if not args.no_save:
        if args.control:
            if args.fine_tune:
                result.append(('fine_tune', args.threshold, first_fitness, second_fitness))
            else:
                result.append(('control', args.threshold, first_fitness, second_fitness))
        else:
            if not args.cifar_svhn:
                result.append(('pathnet', args.threshold, first_fitness, second_fitness))
            else:
                if args.cifar_first:
                    result.append(('pathnet_cifar_first', args.threshold, first_fitness, second_fitness))
                else:
                    result.append(('pathnet_svhn_first', args.threshold, first_fitness, second_fitness))

        if not args.cifar_svhn:
            f = open('./result/result_mnist.pickle', 'wb')
        else:
            f = open('./result/result_cifar_svhn.pickle', 'wb')
        pickle.dump(result, f)
        f.close()

if __name__ == '__main__':
    main()
