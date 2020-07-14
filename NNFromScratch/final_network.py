import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('GALEX_data-extended-feats_new.csv')

x = (data.iloc[:,1:24]).to_numpy()
y = (data.iloc[:,0]).to_numpy()
y = y.reshape(y.shape[0],1)
x = x.astype('float32')
X_train,X_val,y_train,y_val = train_test_split(x,y,test_size = 0.2,random_state = 1213)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
y_train = np.reshape(y_train,(y_train.shape[0],))
y_val = np.reshape(y_val,(y_val.shape[0],))

vecy_train = np.zeros((y_train.shape[0],3))
for i in range(y_train.shape[0]):
  vecy_train[i,y_train[i]] = 1 
  
vecy_val = np.zeros((y_val.shape[0],3))
for i in range(y_val.shape[0]):
  vecy_val[i,y_val[i]] = 1

def sigmoid(array):
  return (1 / (1 + np.exp(-array)))

def sigmoid_derivative(array):
  return sigmoid(array) * (1 - sigmoid(array))

def softmax(array):
  numerator   = np.exp(array)
  denominator = np.sum(numerator,axis = 1,keepdims=True)
  return numerator / denominator

layers = [23,20,40,3]

linear_Z    = {}
nonlinear_A = {}
weights     = {}
gradients   = {}
noise = {}

def initialize_layers(layers):
  for i in range(0,len(layers)-1):
    if(i == 0):
      weights['W'+str(i+1)] = np.random.randn(layers[0],layers[1])
      noise['b'+str(i+1)] = np.random.randn(layers[1]) * 0.1
    else:
      weights['W'+str(i+1)] = np.random.randn(layers[i],layers[i+1])
      noise['b'+str(i+1)] = np.random.randn(layers[i+1]) * 0.1


def linear_nonlinear_activation(layers,X_train,weights,linear_Z,nonlinear_A):
  for i in range(0,len(layers)-2):
    if(i == 0):
      noise['b'+str(i+1)] = np.random.randn(layers[1]) * 0.1
      linear_Z['Z'+str(i+1)] = np.dot(X_train, weights["W1"]) + noise['b1']
      nonlinear_A['A'+str(i+1)] = sigmoid(linear_Z['Z' + str(i+1)])

    else:
      noise['b'+str(i+1)] = np.random.randn(layers[i+1]) * 0.1
      linear_Z['Z'+str(i+1)] = np.dot(nonlinear_A['A'+str(i)],weights['W'+str(i+1)]) + noise['b'+str(i+1)]
      nonlinear_A['A'+str(i+1)] = sigmoid(linear_Z['Z' + str(i+1)])


def backward(X_train,dZ3,weights,gradients,linear_Z,nonlinear_A,layers):
  
  
  for i in range(len(layers)-1,1,-1):
    if(i == len(layers)-1):
      dZ_X = np.dot(dZ3,weights["W"+str(i)].T) * sigmoid_derivative(linear_Z["Z"+ str(i-1)])
      gradients["dW"+str(i-1)] = np.dot(nonlinear_A["A"+str(i-2)].T,dZ_X)
      
    elif(i == 2):
      dZ_X = np.dot(dZ_X,weights["W"+str(i)].T) * sigmoid_derivative(linear_Z["Z"+ str(i-1)])
      gradients["dW"+str(i-1)] = np.dot(X_train.T,dZ_X)
      
    else:
      dZ_X = np.dot(dZ_X,weights["W"+str(i)].T) * sigmoid_derivative(linear_Z["Z"+ str(i-1)])
      gradients["dW"+str(i-1)] = np.dot(nonlinear_A["A"+str(i-2)].T,dZ_X)

def adam(weights, gradients, first_moment, second_moment, time_step, learning_rate, beta1, beta2, smoothing_term=1e-8):

    
    first_moment_bias_correction = {}  
    second_moment_bias_correction = {}           
    
    for i in range(1,len(weights)+1):

        first_moment["dW" + str(i)] = beta1 * first_moment["dW" + str(i)] + (1 - beta1) * gradients['dW' + str(i)]
        first_moment_bias_correction["dW" + str(i)] = first_moment["dW" + str(i)] / (1 - np.power(beta1, time_step))

     
        second_moment["dW" + str(i)] = beta2 * second_moment["dW" + str(i)] + (1 - beta2) * np.power(gradients['dW' + str(i)], 2)
        second_moment_bias_correction["dW" + str(i)] = second_moment["dW" + str(i)] / (1 - np.power(beta2, time_step))
     
    
        numerator = first_moment_bias_correction["dW" + str(i)]
        denominator  = np.sqrt(second_moment_bias_correction["dW" + str(i)] + smoothing_term)
    
        weights["W" + str(i)] = weights["W" + str(i)] - ((learning_rate * numerator) / (denominator))  
   

    return weights, first_moment, second_moment

def moment_initializer(weights) :

    
    first_moment = {}
    second_moment = {}

    for i in range(1,len(weights)+1):
        first_moment["dW" + str(i)] = np.zeros((weights["W" + str(i)].shape[0],weights["W" + str(i)].shape[1]))

        second_moment["dW" + str(i)] = np.zeros((weights["W" + str(i)].shape[0],weights["W" + str(i)].shape[1]))

    
    return first_moment, second_moment

initialize_layers(layers)

first_moment, second_moment = moment_initializer(weights)


adam_params ={'beta1':0.9,
              'beta2':0.99,
              'smoothing_term':1e-8,
              'learning_rate':0.0007,
              'time_step':0,
              'first_moment':first_moment,
              'second_moment':second_moment}

def train_acc():
  c = 0
  linear_nonlinear_activation(layers,X_train,weights,linear_Z,nonlinear_A)
  
  linear_Z['Z'+str(len(weights))] = np.dot(nonlinear_A["A"+str(len(weights)-1)], weights['W'+str(len(weights))]) + noise['b'+str(len(noise))]
  nonlinear_A["A"+str(len(weights))] = softmax(linear_Z['Z'+str(len(weights))])
    
  
  for i in range(X_train.shape[0]):
    if(np.argmax(nonlinear_A["A"+str(len(weights))][i]) == y_train[i]):
      c = c+1
  return c/X_train.shape[0], np.sum(-vecy_train * np.log(nonlinear_A['A'+str(len(weights))]))

def val_acc():
  c = 0
  linear_nonlinear_activation(layers,X_val,weights,linear_Z,nonlinear_A)
  
  linear_Z['Z'+str(len(weights))] = np.dot(nonlinear_A["A"+str(len(weights)-1)], weights['W'+str(len(weights))])
  nonlinear_A["A"+str(len(weights))] = softmax(linear_Z['Z'+str(len(weights))])
    
  
  for i in range(X_val.shape[0]):
    if(np.argmax(nonlinear_A["A"+str(len(weights))][i]) == y_val[i]):
      c = c+1
  return c/X_val.shape[0], np.sum(-vecy_val * np.log(nonlinear_A['A'+str(len(weights))]))

cost_function_train = []
cost_function_val = []

def train(layers,X_train,vecy_train,weights,gradients,linear_Z,nonlinear_A,adam_params,cost_function_train,cost_function_val,total_epochs):
  
  for k in range(total_epochs):
  
    linear_nonlinear_activation(layers,X_train,weights,linear_Z,nonlinear_A)
    
    
    linear_Z['Z'+str(len(weights))] = np.dot(nonlinear_A["A"+str(len(weights)-1)], weights['W'+str(len(weights))]) + noise['b'+str(len(noise))]
    nonlinear_A["A"+str(len(weights))] = softmax(linear_Z['Z'+str(len(weights))])
      
      
    dZ3 = nonlinear_A["A"+str(len(weights))] - vecy_train
    gradients["dW"+str(len(weights))] = np.dot(nonlinear_A["A"+str(len(weights)-1)].T, dZ3)
    backward(X_train,dZ3,weights,gradients,linear_Z,nonlinear_A,layers)
    
    adam_params["time_step"] = adam_params["time_step"] + 1
    
    
    weights, adam_params["first_moment"], adam_params["second_moment"] = adam(weights, gradients, 
                                                                         adam_params["first_moment"], adam_params["second_moment"],
                                                                         adam_params["time_step"], adam_params["learning_rate"],
                                                                         adam_params["beta1"],adam_params["beta2"],
                                                                         adam_params["smoothing_term"])

    if k % 100 == 0:
        print("Epoch %d/%d "%(k,total_epochs),end = ' ')
        train_accuracy,train_loss = train_acc()
        val_accuracy,val_loss = val_acc()
        print('Loss: %f - train accuracy: %f - val_loss: %f - test/val accuracy: %f ' % (train_loss , train_accuracy ,val_loss,val_accuracy))
        cost_function_train.append(train_loss)
        cost_function_val.append(val_loss)

  return cost_function_train,cost_function_val

loss_train,loss_val = train(layers,X_train,vecy_train,weights,gradients,linear_Z,nonlinear_A,adam_params,cost_function_train,cost_function_val,
                     total_epochs = 2301)