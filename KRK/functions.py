import tensorflow as tf
from tensorflow.keras.utils import to_categorical 

from tensorflow import keras
from tensorflow.keras import layers

import autograd.numpy as np
from autograd import grad

import matplot.pyplot as plt

#defining model and cost functions

def model(x,w):
    a = w[0] + np.dot(x,w[1:])
    return a

def linear(w,Xtrain,Ytrain):
    size = float(np.size(Ytrain))
    cost = np.sum(np.abs(model(Xtrain.to_numpy(),w) - Ytrain))
    return cost/size

def perceptron(w,Xtrain, Ytrain):
    size = float(np.size(Ytrain))
    cost = np.sum(np.maximum(0, -Ytrain*model(Xtrain.to_numpy(),w)))
    return cost/size

def softmax(w, Xtrain, Ytrain):
    size = float(np.size(Ytrain))
    cost = np.sum(np.log(1+np.exp(-Ytrain*model(Xtrain.to_numpy(),w))))
    return cost/size
    
def sigmoid(w, Xtrain, Ytrain):
    size = float(np.size(Ytrain))
    cost = np.sum(1/(1+np.exp(-Ytrain*model(Xtrain.to_numpy(),w))))
    return cost/size
    
def tanh(w,Xtrain, Ytrain):
    size = float(np.size(Ytrain))
    cost = np.sum(np.abs(np.tanh(model(Xtrain.to_numpy(),w))-Ytrain))
    return cost/size
    
def grad_descent(g,alpha,max_its,w):
    gradient = grad(g)
    
    weight_history = [w]
    cost_history = [g(w)]
    for k in range(max_its):
        grad_eval = gradient(w)
        
        w = w- alpha*grad_eval
        
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history, cost_history

#Detect function miscounts
def miscount(w,x,y):
    mis = np.argmax(model(x,w),axis = 1)
    counts = []
    for i in range(18):
        bins = np.where(y==i)
        y_s = y[bins]
        mis_s = mis[bins]
        new_count = (mis_s != y_s)
        miscounts = np.sum(new_count)/len(y_s)
        counts.append(miscounts)
    return counts


def compare(Xtrain, Ytrain, Xtest,Y_nothot, alpha, alpha2, max_its, comparison, show = False ):
    #Initialize functions
    g_linear = lambda w:linear(w)
    g_perceptron = lambda w:perceptron(w)
    g_softmax = lambda w:softmax(w)
    g_sigmoid = lambda w:sigmoid(w)
    g_tanh = lambda w:tanh(w)

    w_train = 0.1*np.random.randn(Xtrain.shape[1]+1,Ytrain.shape[1])

    #Determine training weights and costs for each function via grad descent
    lin_train_weight, lin_train_cost = grad_descent(g_linear,alpha,max_its,w_train)
    perc_train_weight, perc_train_cost = grad_descent(g_perceptron,alpha2,max_its, w_train)
    soft_train_weight, soft_train_cost = grad_descent(g_softmax, alpha2,max_its,w_train)
    sig_train_weight, sig_train_cost = grad_descent(g_sigmoid,alpha2,max_its,w_train)
    tanh_train_weight, tanh_train_cost = grad_descent(g_tanh,alpha,max_its,w_train)

    best_cost_lin = np.argmin(lin_train_cost)
    best_weight_lin = lin_train_weight[best_cost_lin]
    misclass_lin = miscount(best_weight_lin,Xtest.to_numpy(),Y_nothot.to_numpy()[:,0])

    best_cost_perc = np.argmin(perc_train_cost)
    best_weight_perc = perc_train_weight[best_cost_perc]
    misclass_perc = miscount(best_weight_perc,Xtest.tonumpy(),Y_nothot.to_numpy()[:,0])

    best_cost_soft = np.argmin(soft_train_cost)
    best_weight_soft = soft_train_weight[best_cost_soft]
    misclass_soft = miscount(best_weight_soft,Xtest.to_numpy(),Y_nothot.to_numpy()[:,0])

    best_cost_sig = np.argmin(sig_train_cost)
    best_weight_sig = sig_train_weight[best_cost_sig]
    misclass_sig = miscount(best_weight_sig,Xtest.to_numpy(),Y_nothot.to_numpy()[:,0])

    best_cost_tanh = np.argmin(tanh_train_cost)
    best_weight_tanh = tanh_train_weight[best_cost_tanh]
    misclass_tanh = miscount(best_weight_tanh,Xtest.to_numpy(),Y_nothot.to_numpy()[:,0])


    if comparison == 'cost':
        plt.figure()
        plt.title('Cost Histories for Training Set with Various Functions')
        plt.plot(lin_train_cost, color ='red', label = 'linear')
        plt.plot(perc_train_cost, color = 'orange',label = 'perceptron')
        plt.plot(soft_train_cost, color = 'blue', label = 'softmax')
        plt.plot(sig_train_cost, color = 'green', label ='sigmoid')
        plt.plot(tanh_train_cost, color = 'purple',label = 'tanh')
        plt.xlabel('Iterations')
        plt.ylabel('Cost')
        plt.legend()
        
        if show:
            plt.show()
    
    elif comparison == 'miscount':
        k = np.arange(0,18)
        plt.figure()
        plt.title('Misclassifications')
        plt.plot(k,misclass_lin, label = ' linear')
        plt.plot(k,misclass_perc, label = 'perceptron')
        plt.plot(k,misclass_soft, label = 'softmax')
        plt.plot(k,misclass_sig, label = 'sigmoid')
        plt.plot(k, misclass_tanh, label = 'tanh')
        plt.legend()
        plt.xlabel('Depth-of-win')
        plt.ylabel('Percentage')
        
        if show:
            plt.show()


#Function for learning positions to minimize depth of win
def learning_model(Xtrain, Ytrain, Xtest, Ytest, batch_size, epochs, activation):

    model = tf.keras.models.Sequential([
                                        tf.keras.Input(shape = (Xtrain.shape[1],)),
                                        tf.keras.layers.Dense(50,activation = 'sigmoid'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(30,activation = 'sigmoid'),
                                        tf.keras.layers.BatchNormalization(),
                                        tf.keras.layers.Dense(18, activation = activation)
                                        ])
    tf.keras.optimizers.SGD(learning_rate = 0.01)
    model.compile(optimizer = 'SGD', loss = 'categorical_crossentropy',metrics = ['accuracy'])

    history = model.fit(Xtrain, Ytrain,
                        validation_data = (Xtest,Ytest),
                        batch_size = batch_size,
                        epochs = epochs,
                        verbose = 1,
                        shuffle = True)
    return history

#function to train model with different activation functions and plot comparisons
def activation_comparison(Xtrain,Ytrain,Xtest,Ytest,batch_size,epochs,act_func, show = False):    #act_func is a list of all activation functions to compare
    
    histories = {}

    for act in act_func:
        hist = f'hist_{act}'
        hist_func = learning_model(Xtrain,Ytrain,Xtest,Ytest,batch_size,epochs,act)
        
        histories[hist] = hist_func
    if show == True:
        
        fig, ax = plt.subplots(figsize = (10,10))
        ax.set_title('Accuracy of Models')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Accuracy')
        for key in histories:
            ax.plot(histories[key].history['val_accuracy'],label = str(key))
        plt.legend()
        plt.show()
        
        
        fig, ax = plt.subplots(figsize = (10,10))
        ax.set_title('Loss of Models')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        for key in histories:
            ax.plot(histories[key].history['val_loss'],label = str(key))
        plt.legend()
        plt.show()
        
