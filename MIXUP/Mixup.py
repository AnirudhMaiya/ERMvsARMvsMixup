import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

data = pd.read_csv('GALEX_data-extended-feats_new.csv')

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

def mixup_data(x, y, alpha=1.0):
    #SOURCE FAIR https://github.com/facebookresearch/mixup-cifar10
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)
    mixed_x = lam * x + (1 - lam) * x[index,:]   #ik i should have used MINMAX SCALER (0,1)
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def accuracy(y_hat,y_true,lam,is_train):
  if is_train:
    labels1,labels2 = y_true[0],y_true[1]
    y_hat = F.softmax(y_hat,dim = 1)
    _, predicted = torch.max(y_hat, 1)
    a_is = lam * predicted.eq(labels1).cpu().sum()
    b_is = (1 - lam) * predicted.eq(labels2).cpu().sum()
    total_correct = a_is + b_is
    return total_correct
  else:
     y_hat = F.softmax(y_hat,dim = 1)
     _, predicted = torch.max(y_hat, 1)
     total_correct = (predicted.reshape(-1,1) == y_true.reshape(-1,1)).sum().item()
     return total_correct

def train(model,epochs,loader,alpha_is):
  model.train()
  correct = 0
  cc = 0
  loss_list = []
  for idx,(i,j) in enumerate(loader):
    inputs,labels = i.to(device),j.to(device)
    inputs,labels1,labels2,lamda = mixup_data(inputs,labels,alpha = alpha_is)
    opt.zero_grad()
    outputs = model(inputs)
    loss_func = mixup_criterion(labels1, labels2, lamda)
    loss_is = loss_func(criterion, outputs)
    loss_is.backward()
    opt.step()
    loss_list.append(loss_is.item())
    tot_labels = [labels1,labels2]
    correct = correct + accuracy(outputs,tot_labels,lamda,True)
  mean_loss = sum(loss_list)/len(loss_list)
  mean_acc = (correct/len(loader.dataset))*100
  print("[%d/%d] Training Accuracy : %f and Train Loss : %f "%(epochs,total_epochs,mean_acc,mean_loss))
  return mean_loss,mean_acc

def test(model,epochs,loader):
  model.eval()
  correct = 0
  loss_list = []
  with torch.no_grad():
    for i,j in loader:
      inputs,labels = i.to(device),j.to(device)
      outputs = model(inputs)
      loss_is = criterion(outputs,labels)
      correct = correct + accuracy(outputs,labels,_,False)
      loss_list.append(loss_is.item())
    mean_loss = sum(loss_list)/len(loss_list)
    mean_acc = (correct/len(loader.dataset))*100
    print("[%d/%d] Test Accuracy : %f and Test Loss : %f"%(epochs,total_epochs,mean_acc,mean_loss))
    print('---------------------------------------------------------------------')
  return mean_loss,mean_acc

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
criterion = nn.CrossEntropyLoss()
opt = torch.optim.SGD(net.parameters(),lr = 0.1,momentum = 0.9)

total_epochs = 200
train_loss = []
train_acc = []
test_loss = []
test_acc = []
alpha_is = 1
for s in range(1,total_epochs + 1):
  a,b = train(net,s,trainloader,alpha_is)
  c,d = test(net,s,testloader)
  train_loss.append(a)
  train_acc.append(b)
  test_loss.append(c)
  test_acc.append(d)