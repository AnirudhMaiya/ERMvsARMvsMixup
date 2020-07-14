import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

data = pd.read_csv('GALEX_data-extended-feats_new.csv')

x = (data.iloc[:,1:24]).to_numpy()
y = (data.iloc[:,0]).to_numpy()
class_weight = compute_class_weight('balanced',list(np.unique(y)), list(y)).astype('float32')
y = y.reshape(y.shape[0],1)
x = x.astype('float32')

X_train,X_val,y_train,y_val = train_test_split(x,y,test_size = 0.2,random_state = 1213)
class_weight = compute_class_weight('balanced',list(np.unique(y_train.reshape(-1,))), list(y_train.reshape(-1,))).astype('float32')
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
y_train = np.reshape(y_train,(y_train.shape[0],))
y_val = np.reshape(y_val,(y_val.shape[0],))

X_train,y_train,X_val,y_val = torch.tensor(X_train),torch.tensor(y_train),\
                              torch.tensor(X_val),torch.tensor(y_val)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset = torch.utils.data.TensorDataset(X_val, y_val)
trainloader = torch.utils.data.DataLoader(train_dataset, batch_size =128, pin_memory=True,shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset,batch_size =2048,pin_memory=True,shuffle = False)

#source Zico Kolter's tutorial
def pgd_inf(model,inputs,labels,epsilon,alpha,iterations):
  delta = torch.rand_like(inputs,requires_grad = True).to(device)
  for i in range(iterations):
    outputs = model(inputs+delta)
    loss_Q = loss(outputs,labels.reshape(-1,))
    loss_Q.backward()
    delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
    delta.grad.zero_()
  return delta.detach()

def accuracy(y_hat,y_true):
  y_hat = F.softmax(y_hat,dim = 1)
  _, predicted = torch.max(y_hat, 1)
  total_correct = (predicted.reshape(-1,1) == y_true.reshape(-1,1)).sum().item()
  return total_correct

def train(model,epochs,loader,iterations,alpha,epsilon):
  model.train()
  correct = 0
  loss_list = []
  for i,j in loader:
    inputs,labels = i.to(device),j.to(device)
    adv_eg = pgd_inf(model,inputs,labels,epsilon,alpha,iterations)
    opt.zero_grad()
    outputs = model(inputs+adv_eg)
    loss_is = loss(outputs,labels.reshape(-1,))
    loss_is.backward()
    opt.step()
    loss_list.append(loss_is.item())
    correct = correct + accuracy(outputs,labels)
  
  print("[%d/%d] Training Accuracy : %f"%(epochs,total_epochs, (correct/len(loader.dataset)) * 100))
  return sum(loss_list)/len(loss_list),(correct/len(loader.dataset)) * 100

def test(model,epochs,loader):
  model.eval()
  correct = 0
  with torch.no_grad():
    for i,j in loader:
      inputs,labels = i.to(device),j.to(device)
      outputs = model(inputs)
      correct = correct + accuracy(outputs,labels)
    print("[%d/%d] Test Accuracy : %f"%(epochs,total_epochs,(correct/len(loader.dataset))*100))
    print('---------------------------------------------------------------------')
  return (correct/len(loader.dataset)) * 100

class Network(nn.Module):
  def __init__(self,inp_shape):
    super(Network,self).__init__()
    self.l1 = nn.Linear(inp_shape,20)
    self.l2 = nn.Linear(20,40)
    self.l3 = nn.Linear(40,3)

  def forward(self,x):
    x = F.tanh(self.l1(x))
    x = F.tanh(self.l2(x))
    x = self.l3(x)
    return x

class FocalLoss(nn.Module):
  def __init__(self,gamma,eps):
    super(FocalLoss,self).__init__()
    self.gamma = gamma
    self.eps = eps

  def forward(self,y_hat,y_true):
    one_hot = torch.eye(NUM_CLASSES).to(device)
    y_true = one_hot[y_true]
    y_hat = torch.clamp(F.softmax(y_hat,dim = 1),self.eps,1 - self.eps)
    loss = -1 * ((1 - y_hat)**self.gamma) * y_true * torch.log(y_hat)
    return loss.mean()

NUM_CLASSES = 3
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(12)
net = Network(X_train.shape[1]).to(device)
class_weights = torch.tensor(class_weight.copy()).to(device)
#loss = nn.CrossEntropyLoss()
loss = FocalLoss(gamma = 2,eps = 1e-7)
opt = torch.optim.SGD(net.parameters(),lr = 0.1,momentum = 0.9)

total_epochs =200
train_loss = []
train_acc = []
test_acc = []
for s in range(1,total_epochs + 1):
  test_on_adv_eps(net,s,testloader,iterations = 10,alpha = 1e-1,epsilon = 0.05)
  a,b = train(net,s,trainloader,iterations = 10,alpha = 1e-1,epsilon = 0.05)
  c = test(net,s,testloader)
  train_loss.append(a)
  train_acc.append(b)
  test_acc.append(c)
  if s >=100:
    for ui in opt.param_groups:
      ui['lr'] = 0.01