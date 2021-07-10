from torch.utils.data import DataLoader
import dataset_prep, models 
from time import time
from torch import optim
import torch
import numpy as np

path_init = 'Pickle/data.p'

#Usa GPU se estiver disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = dataset_prep.data('train', models.args.fold, path_init, dataset_prep.aug_transforms)
test_dataset = dataset_prep.data('test', models.args.fold, path_init, dataset_prep.val_transforms)

epochs = models.args.epochs;

def train(train_dataset, val, path, validloader=None):

    for epoch in range(epochs):
        
        print(f"Epoch {epoch+1}/{epochs}.. ")
            
        avg_acc=0;
        avg_prec=0;
        avg_mae=0;
        
        model.train()                                                         #Preparação modelo para o treinamento (training)
        tic = time()                                                          #Os tempos são expressos como números de ponto flutuante
                                                                              #e, retorna o tempo mais preciso disponível
        for inputs, labels in train_ld:
            inputs = inputs.to(device)                                        #Coloca tensor 'inputs' no GPU 
            labels = labels.to(device, torch.int64)
            
            optimizer.zero_grad()                                             #Limpa os gradientes de todas as variáveis otimizadas 
            outputs = model(inputs)                                           #Forward pass: Calcula saídas previstas passando entradas para o modelo
            
            f = f'outputs_train-architecture-{models.args.architecture}-lr-{models.args.lr}-batch_size-{models.args.batch_size}.pth'
            torch.save(outputs, f)
            
            loss = model.loss(outputs,labels)
            loss.backward()                                                   #Backward pass: Calcula o gradiente da perda em relação aos parâmetros do modelo
            optimizer.step()                                                  #Realiza uma única etapa de otimização (atualização de parâmetro)
            
            K_hat = model.softmax(outputs)
            c_pred=model.pred(K_hat)[1]

            avg_acc += model.acc(labels.cpu(),c_pred.cpu())/len(train_ld)
            avg_mae += model.mae(labels.cpu(),c_pred.cpu())/len(train_ld)
            avg_prec += model.prec(labels.cpu(),c_pred.cpu())/len(train_ld)
       
        dt = time() - tic
        
        print('TRAIN RESULTS:\nACCURACY: %.3f\nMAE: %.3f\nPRECISION: %.3f\n'%(avg_acc,avg_mae,avg_prec)) 
        print('Time elapsed: %d minutes and %d seconds'%(int(dt/60),dt%60))

        
        if validloader is not None:
            model.eval()                                                      #Desativa as camadas Dropout e é equivalente a model.train(False)
            val_acc, val_prec, val_mae = test(validloader,True)
            print('TEST RESULTS:\nACCURACY: %.3f\nMAE: %.3f\nPRECISION: %.3f\n'%(val_acc,val_prec,val_mae)) 


            
def test(test_ld):

    with torch.no_grad():
        
        model.eval()                                                             
        avg_acc=0;
        avg_prec=0;
        avg_mae=0;
        Phat = []
    
        for inputs,labels in train_ld:
            inputs =inputs.to(device)
            labels = labels.to(device,torch.int64)
            
            outputs=model(inputs)
            f = f'outputs_test-architecture-{models.args.architecture}-lr-{models.args.lr}-batch_size-{models.args.batch_size}.pth'
            torch.save(outputs, f)
            
            K_hat = model.softmax(outputs)
            Phat += list(K_hat.cpu().numpy())
            c_pred=model.pred(outputs)[1]
            
            avg_acc += model.acc(labels.cpu(),c_pred.cpu())/len(test_ld)
            avg_mae += model.mae(labels.cpu(),c_pred.cpu())/len(test_ld)
            avg_prec += model.prec(labels.cpu(),c_pred.cpu())/len(test_ld)
    
        print('TESTING RESULTS (AVERAGE):\nACCURACY: %.3f\nMAE: %.3f\nPRECISION: %.3f\n'%(avg_acc,avg_mae,avg_prec)) 
        prefix = f'architecture-{models.args.architecture}-lr-{models.args.lr}-batch_size-{models.args.batch_size}'
        np.savetxt('output-' + prefix + '-proba.txt', Phat)
              
        return avg_acc,avg_prec,avg_mae        
        
    
    
if __name__ == '__main__':     
  train_ld, test_ld = DataLoader(train_dataset, models.args.batch_size, shuffle=True, num_workers=2), DataLoader(test_dataset, models.args.batch_size, shuffle=False, num_workers=2)
  model = models.Net(models.args.architecture)
  model = model.to(device)                                                   
  optimizer = optim.Adam(model.parameters(),models.args.lr)
  path = '/Resultados'
  train(train_ld, test_ld, path)
