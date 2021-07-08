from sklearn import metrics
import torch.nn as nn
import argparse 
import numpy as np
import torch
import os
import train

path_init = train.path_init

class metricas(nn.Module):

  def accuracy(y, yhat):
      return metrics.accuracy_score(y, yhat)

  def recall(y, yhat):
      return metrics.recall_score(y, yhat)

  def precision(y, yhat):
      return metrics.precision_score(y, yhat)

  def mae(y, yhat):
      return metrics.mean_absolute_error(y, yhat)

  def auc(y, yhat):
    yhat = yhat / yhat.sum(1, keepdims=True)
    return metrics.roc_auc_score(y, yhat, multi_class='ovo')

  def loss(yhat, y):
    return criterion(yhat, y)


YY = [pickle.load(open(train.path_init, 'rb'))[fold]['test'][1] for fold in range(5)]

def evaluate(YY, outputs, architecture=args.architecture):
  resultados = []

    for fold in range(5):
        Y = YY[fold]
        start = f'output-architecture-{architecture}-fold-{fold}-batchsize-{args.batchsize}-dropout-{args.dropout}-lr-{args.lr}'
        filename = [o for o in outputs if o.split('/')[-1].startswith(start)]
        assert len(filename), f'Empty filenames starting with {start}'
        Yhat = np.loadtxt(filename[0], delimiter=',')

        resultados.append(metricas.auc(Y, Yhat))
        resultados.append(metricas.mae(Y, Yhat))
        resultados.append(metricas.acc(Y, Yhat))
        resultados.append(metricas.prec(Y, Yhat))
        resultados.append(metricas.recall(Y, Yhat))
        resultados.append(metricas.loss(Yhat, Y))

    return resultados





