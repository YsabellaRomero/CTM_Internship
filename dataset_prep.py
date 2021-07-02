from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pickle
import matplotlib.pyplot as plt


class data(Dataset):
  def __init__(self, fase, fold, path, transform=None):                         #type --> se é treino ou teste
    self.inputs, self.labels = pickle.load(open(path, 'rb'))[fold][fase]        #transform --> se vamos usar transformações de treino ou teste
    self.transform = transform                                                  #fold --> faz chamada para um dos folds

  def __getitem__(self,index):
    filename = self.inputs[index]
    inputs = Image.open(filename)
    inputs = self.transform(inputs)
    labels = self.labels[index]
    return inputs, labels

  def __len__(self):
    return len(self.inputs)

aug_transforms = transforms.Compose([                                                                                              
    transforms.Resize((224,224)),                                               #Redimensionamento da imagem
    transforms.RandomAffine(180, (0, 0.1), (0.9, 1.1)),                          #Roda a imagem e faz translações de forma aleatório
    transforms.RandomHorizontalFlip(),                                          #Inverte a imagem da direita para a esquerda e vice-versa de forma alteatória
    transforms.RandomVerticalFlip(),                                            #Inverte a imagem de cima para baixo e vice-versa de forma alteatória
    transforms.ColorJitter(saturation=(0.5, 2.0)),                              #Altera aleatoriamente o brilho, a saturação e outras propriedades da imagem
    transforms.ToTensor(),  # vgg normalization                                 #Mandar para tensor
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))          #Esta normalização tem a ver, em si, com a arquitetura --> quando usamos
])                                                                              #transfer learning, as arquiteturas foram testadas com outro tipos de  
                                                                                #imagens e elas depois foram normalizadas com estes valores a serem usados

val_transforms = transforms.Compose([  
    transforms.Resize((224,224)), 
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


path = 'Pickle/data.p'

train_dataset = data('train', 0, path, aug_transforms)

for inputs, labels in train_dataset:
    print(inputs)

  


