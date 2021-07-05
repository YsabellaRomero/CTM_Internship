from torch.utils.data import DataLoader
import dataset_prep, models
import argparse 

parser = argparse.ArgumentParser()                                            #Criação de um objeto ArgumentParsec
parser.add_argument('architecture', choices=['alexnet', 'densenet161',        #Lista de arquiteturas a treinar
    'googlenet', 'inception_v3', 'mnasnet1_0', 'mobilenet_v2', 'resnet18',
    'resnext50_32x4d', 'shufflenet_v2_x1_0', 'squeezenet1_0', 'vgg16',
    'wide_resnet50_2'])
parser.add_argument('method', choices=['Net'])                                

parser.add_argument('fold', type=int, choices=range(8))
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-3)
args = parser.parse_args()

train_dataset = dataset_prep.data('train', args.fold, dataset_prep.path, dataset_prep.aug_transforms)
train = DataLoader(train_dataset, args.batchsize, True)

test_dataset = dataset_prep.data('test', args.fold, dataset_prep.path, dataset_prep.val_transforms)
test = DataLoader(test_dataset, args.batchsize, True)



