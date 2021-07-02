import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

criterion = nn.CrossEntropyLoss()

class Net(nn.Module):
    def __init__(self, pre_trained, num_outputs):
        super().__init__()
        self.n_labels = num_outputs
        classifier = getattr(models, pre_trained, 'pretrained attribute not found')             #if 'pretrained' attribute is found, its value will
        self.classifier = nn.Sequential(                                                        #be printed, otherwise 3rd parameter will be returned
            nn.Linear(512 * 7 * 7, 512),                                                        #Last dimension from pooling layer is 7x7x512 in VGG16
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_outputs)
        )

    def foward(self, x):
        return self.classifier(x)
    
    def loss(self, outputs, labels):
        return criterion(outputs, labels)

    def softmax(self, outputs):
        return F.softmax(outputs, dim=1)
        
    
n_labels = 4