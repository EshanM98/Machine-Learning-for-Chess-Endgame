import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from functions import *


dataset_url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/chess/king-rook-vs-king/krkopt.data'

#Original dataset has columns (a,1), (b,3) and (c,2) which provides coordinates of the White King, White Rook, and Black King, respectively
#Letter columns provide the column position and number columns provide the row positions
dataset = pd.read_csv(dataset_url)

#Converting letter coordinates to int values
letters = ['a','b','c']
for letts in letters:
    dataset[letts] = dataset[letts].map(lambda x:ord(x)-96).astype(np.int64)

#Convert depth-of-wins (draw column) string to int
dataset['draw'] = pd.factorize(dataset['draw'])[0]



#Create training and test sets
train, test = train_test_split(dataset, test_size=0.25)
Xtrain = pd.DataFrame(train, columns = ['a','1','b','3','c','2'])
Ytrain = pd.DataFrame(train, columns = ['draw'])
Xtest = pd.DataFrame(test, columns = ['a','1','b','3','c','2'])
Ytest = pd.DataFrame(test, columns = ['draw'])
Y_nothot = pd.DataFrame(test,columns = ['draw'])
Ytrain = tf.keras.utils.to_categorical(Ytrain.astype('int64'),18)
Ytest = tf.keras.utils.to_categorical(Ytest.astype('int64'),18)


alpha = 0.1
alpha2 = 1
max_its = 100

compare(Xtrain, Ytrain, Xtest, Y_nothot, alpha, alpha2, max_its, comparison = 'cost', show = True )

compare(Xtrain, Ytrain, Xtest, Y_nothot, alpha, alpha2, max_its, comparison = 'miscount', show = True )


activations = ['softmax','relu','sigmoid','tanh']
batch_size = 64
epochs = 100

activation_comparison(Xtrain,Ytrain,Xtest,Ytest,batch_size,epochs,act_func = activations, show = True)


