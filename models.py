import torch.nn as nn
import torch
import torch.nn.functional as F
from torchvision import models

criterion = nn.CrossEntropyLoss()

class Net(nn.Module):
    def __init__(self, pretrained, num_outputs):
        super().__init__()
        self.n_labels = num_outputs
        model = models.vgg16(pretrained=True)                                           #Instanciar um modelo pré-treinado baixará seus pesos para um diretório de cache
        model = nn.Sequential(*list(model.children())[:-1])                             #Novo modelo criado através da lista com todas as camadas (à exceção da última) do modelo pré-treinado                            
        last_dimension = torch.flatten(model(torch.randn(1, 3, 224, 224))).shape[0]      #Dimensão da última camada do modelo     
        self.model = nn.Sequential(                                                     #torch.randn(1, 3, 224, 224) --> tensor com dimensão 1x3x224x224 
            model, 
            nn.Flatten(),                                                               #Aplana o tensor 
            nn.Dropout(0.2),                                                            #Dropout é uma técnica que seleciona 'neurals' aleatoriamente e os ignora
            nn.Linear(last_dimension, 512),                                                      
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )

    def foward(self, x):
        return self.model(x)
    
    def loss(self, outputs, labels):
        return criterion(outputs, labels)

    def softmax(self, outputs):
        return F.softmax(outputs, dim=1)
        
    def pred(self, pred):
        return torch.max(pred,1)
        
    
n_labels = 4
Net(True, n_labels)