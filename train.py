from torch.utils.data import DataLoader
import dataset_prep, models 
from time import time
from torch import optim
from sklearn import metrics
import torch
import numpy as np
import os

path_init = 'Pickle/data.p'

#Usa GPU se estiver disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = dataset_prep.data('train', 4, path_init, dataset_prep.aug_transforms)
test_dataset = dataset_prep.data('test', 4, path_init, dataset_prep.val_transforms)

epochs = 10;

def train(train_dataset, val, path, validloader=None):
    
    train_acc=[]
    train_bal_acc=[]
    train_loss=[]
    tval_acc=[]
    tval_bal_acc=[]
    tval_loss=[]
    
    for epoch in range(epochs):
        
        print(f"Epoch {epoch+1}/{epochs}.. ")
            
        bal_acc=0;
        avg_acc=0;
        avg_loss=0;
        
        model.train()                                                         #Preparação modelo para o treinamento (training)
        tic = time()                                                          #Os tempos são expressos como números de ponto flutuante
                                                                              #e, retorna o tempo mais preciso disponível
        for inputs, labels in train_ld:
            inputs = inputs.to(device)                                        #Coloca tensor 'inputs' no GPU 
            labels = labels.to(device, torch.int64)
            
            optimizer.zero_grad()                                             #Limpa os gradientes de todas as variáveis otimizadas 
            outputs = model(inputs)                                           #Forward pass: Calcula saídas previstas passando entradas para o modelo
            
            loss = model.loss(outputs,labels)
            loss.backward()                                                   #Backward pass: Calcula o gradiente da perda em relação aos parâmetros do modelo
            optimizer.step()                                                  #Realiza uma única etapa de otimização (atualização de parâmetro)
            
            c_pred=model.pred(outputs)[1]

            avg_acc += metrics.accuracy_score(labels.cpu(),c_pred.cpu())/len(train_ld)               #Pontuação de classificação de precisão
            avg_loss += loss/len(train_ld)
            bal_acc += metrics.balanced_accuracy_score(labels.cpu(),c_pred.cpu())/len(train_ld)      #Cálculo da precisão balanceada
       
        dt = time() - tic
        
        print('TRAINING RESULTS:\nACCURACY: %.2f BAL_ACCURACY: %.2f LOSS: %.3f'%(avg_acc,bal_acc,avg_loss))
        print('Time elapsed: %d minutes and %d seconds'%(int(dt/60),dt%60))
        
        train_acc += [avg_acc]
        train_bal_acc += [bal_acc]
        train_loss += [avg_loss]
        
        if validloader is not None:
            model.eval()
            val_acc,val_loss,val_bal_acc=test(validloader,True)
            tval_acc+=[val_acc]
            tval_bal_acc+=[val_bal_acc]
            tval_loss+=[val_loss]
            
    np.savez(os.path.join(path,'training.npz'),train_acc=train_acc,train_bal_acc=train_bal_acc,train_loss=train_loss) 
    
    if validloader is not None:       
        np.savez(os.path.join(path,'validation.npz'),tval_acc=tval_acc,tval_bal_acc=tval_bal_acc,tval_loss=tval_loss)

            
def test(test_ld, val=False):
    model.eval()                                                              #Desativa as camadas Dropout e é equivalente a model.train(False)
    
    avg_acc=0;
    avg_loss=0;
    bal_acc=0

    for inputs,labels in train_ld:
        inputs =inputs.to(device)
        labels = labels.to(device,torch.int64)
        
        outputs=model(inputs)
        loss=model.loss(outputs,labels) 
        
        c_pred=model.pred(outputs)[1]
        
        avg_acc+=metrics.accuracy_score(labels.cpu(),c_pred.cpu())/len(test_ld)
        avg_loss+=loss/len(test_ld)
        bal_acc+=metrics.balanced_accuracy_score(labels.cpu(),c_pred.cpu())/len(test_ld)

    print('TESTING RESULTS:\nACCURACY: %.2f BAL_ACCURACY: %.2f LOSS: %.3f'%(avg_acc,bal_acc,avg_loss)) 
          
    if val:
        return avg_acc,avg_loss,bal_acc         
        
if __name__ == '__main__':     
    
    batch_size = [8, 16, 32, 64];
    model_n = ['vgg16', 'googlenet'];
    lr = [1e-4, 1e-5, 1e-6];
    
    for i in model_n:
        for j in lr:
            for k in batch_size:
                train_ld = DataLoader(train_dataset, k, shuffle=True, num_workers=2)
                test_ld = DataLoader(test_dataset, k, shuffle=False, num_workers=2)
                model = models.Net(i)
                model = model.to(device)                                                    
                optimizer = optim.Adam(model.parameters(),j)
                path = 'Resultados\model_'+str(i)+'_lr_'+str(j)+'_batchsize_'+str(k)
                train(train_ld,path,test_ld,path)