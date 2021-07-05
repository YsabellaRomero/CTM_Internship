from torch.utils.data import DataLoader
import dataset_prep, models
from torch import optimizer
import torch
import argparse 

#Usa GPU se estiver disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def train(train_dataset, epochs=args.epochs, verbose=True):
    
    for epochs in range(epochs):
        if verbose:
            print(f"Epoch {epochs+1}/{epochs}.. ")
            
        model.train()
        train_loss = 0
        valid_loss = 0
        
        #Treino do modelo 
        model.train()                                                         #Preparação modelo para treinamento (training)
        
        for X, Y in train_loader:
            X = X.to(device)
            Y = Y.to(device, torch.int64)
            optimizer.zero_grad()                                             #Limpa os gradientes de todas as variáveis otimizadas 
            Y_hat = model(X)                                                  #Forward pass: Calcula saídas previstas passando entradas para o modelo
            loss = model.loss(Y_hat,Y)
            loss.backward()                                                   #Backward pass: Calcula o gradiente da perda em relação aos parâmetros do modelo
            optimizer.step()                                                  #Realiza uma única etapa de otimização (atualização de parâmetro)
            train_loss += loss.item()*data.size(0)                            #Atualiza running training loss
        

        
    