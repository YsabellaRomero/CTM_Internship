import torch.nn as nn
import torch
import torch.nn.functional as F
import argparse 
from torchvision import models
from sklearn import metrics

criterion = nn.CrossEntropyLoss()

parser = argparse.ArgumentParser()                                            #Criação de um objeto ArgumentParsec

parser.add_argument('--method', default='Net')
parser.add_argument('--architecture', choices=['vgg16', 'googlenet'], default='vgg16') 
parser.add_argument('--fold', type=int, choices=range(5), default=4)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, choices=[8, 16, 32, 64], default=32)
parser.add_argument('--dropout', type=int, choices=[0.2, 0.3, 0.4], default=0.3)
parser.add_argument('--lr', type=float, choices=[1e-4, 1e-5, 1e-6], default=1e-4)
parser.add_argument("-f", "--file", required=False) 
args = parser.parse_args()

class Net(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        model = models.vgg16(pretrained=True)                                           #Instanciar um modelo pré-treinado baixará seus pesos para um diretório de cache
        model = nn.Sequential(*list(model.children())[:-1])                             #Novo modelo criado através da lista com todas as camadas (à exceção da última) do modelo pré-treinado                            
        last_dimension = torch.flatten(model(torch.randn(1, 3, 224, 224))).shape[0]     #Dimensão da última camada do modelo     
        self.model = nn.Sequential(                                                     #torch.randn(1, 3, 224, 224) --> tensor com dimensão 1x3x224x224 
            model, 
            nn.Flatten(),                                                               #Aplana o tensor 
            nn.Dropout(args.dropout),                                                            #Dropout é uma técnica que seleciona 'neurals' aleatoriamente e os ignora
            nn.Linear(last_dimension, 512),                                                      
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(256, 4)
        )

    def forward(self, x):
        return self.model(x)
    
    def loss(self, pred, labels):
        return criterion(pred, labels)

    def softmax(self, outputs):
        return F.softmax(outputs, dim=1)
    
    def pred(self, pred):
        return torch.max(pred,1)
    
    ## MÉTRICAS ##

    def acc(self, y, yhat):                                                   
        return metrics.accuracy_score(y, yhat)

    def prec(self, y, yhat):
        return metrics.precision_score(y, yhat, average='weighted', zero_division=1)

    def mae(self, y, yhat):
        return metrics.mean_absolute_error(y, yhat)
        