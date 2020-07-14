import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('GALEX_data-extended-feats_new.csv')

data.tail()

x = (data.iloc[:,1:24]).to_numpy()
y = (data.iloc[:,0]).to_numpy()
y = y.reshape(y.shape[0],1)
x = x.astype('float32')

print('Classes are : ',*list(np.unique(y)))

X_train,X_val,y_train,y_val = train_test_split(x,y,test_size = 0.2,random_state = 1213)
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

def accuracy(y_hat,y_true):
  y_hat = F.softmax(y_hat,dim = 1)
  _, predicted = torch.max(y_hat, 1)
  total_correct = (predicted.reshape(-1,1) == y_true.reshape(-1,1)).sum().item()
  return total_correct

def train(model,epochs,loader):
  model.train()
  correct = 0
  cc = 0
  loss_list = []
  for i,j in loader:
    inputs,labels = i.to(device),j.to(device)
    opt.zero_grad()
    outputs = model(inputs)
    loss_is = loss(outputs,labels)
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(12)
net = Network(X_train.shape[1]).to(device)
loss = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(),lr = 0.1,momentum = 0.9)

total_epochs = 200
train_loss = []
train_acc = []
test_acc = []
for s in range(1,total_epochs + 1):
  a,b = train(net,s,trainloader)
  c = test(net,s,testloader)
  train_loss.append(a)
  train_acc.append(b)
  test_acc.append(c)