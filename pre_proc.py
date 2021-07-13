import os
from sklearn.model_selection import StratifiedKFold
import pickle
import numpy as np

num_classes=4;
divs=5;
data_dict = []

def loadImages():
    
    folder = "dataset"
    inputs = []
    labels = []
    i = 0
    
    dirList = os.listdir(folder)
    
    for subfolder in dirList:
        
        if subfolder.endswith(".DS_Store"):
            continue

        else:
            sub_path = os.path.join(folder,subfolder)
            subdirList = os.listdir(sub_path)
            
            for file in subdirList:
                
                im_path = os.path.join(sub_path,file)       
            
                if file.endswith(".jpg"):
                    inputs.append(im_path)                   #guarda caminho no vetor inputs  
                    labels.append(i)                         #diz a label que corresponde à imagem
                
                else:
                    continue
            
            i += 1

    return inputs, labels


inputs, labels = loadImages()
inputs, labels = np.array(inputs), np.array(labels)          #transforma de lista para array

skf=StratifiedKFold(n_splits=divs,shuffle=True,random_state=1234);

for tr, ts in skf.split(inputs,labels):
    data_dict.append({'train':(inputs[tr],labels[tr]),'test':(inputs[ts],labels[ts])})

pickle.dump(data_dict,open('Pickle/data.p','wb'))

