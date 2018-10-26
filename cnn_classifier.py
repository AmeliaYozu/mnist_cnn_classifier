#!/usr/local/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model

## Steps:
# 1. Load Data
# 2. Preprocess data (incluing check for null and missing values)
# 3. Split training data into training set and dev set by dev set ~ 4000 ?
# 4. setup model
# 5. train model
# 6. test model by dev set
# *7. tune model (eg. add regularization, change optimizer, try more options on hyperparameters) - can be combined to step  5 ?

################################################
##
# 1. Load Data
#      &
# 2. Preprocess Data
#      &
# 3. Split into Training&Dev Sets
##
################################################

raw_data = pd.read_csv("./train.csv")

# check for null and missing values
raw_data.isnull().any().describe()

# split to training and dev set by 1:9
dev_set = raw_data.sample(frac=0.1)
training_set = raw_data.drop(dev_set.index)

# free some space
del(raw_data) # release RAM

# split into X and y
y_train = training_set['label']
X_train = training_set.drop(columns='label')
# X_train.reshape(X_train.shape[0],28,28)
y_dev  = dev_set['label']
X_dev = dev_set.drop(columns='label')
# X_dev.reshape(X_dev.shape[0],28,28)

# normalization
X_train = X_train/255.0
X_dev = X_dev/255.0

# free some space
del(dev_set,training_set)

# plot counts of splitted training and dev data to make sure they are sampled evenly
# plt.figure()
# plt.subplot(2,1,1)
# g = sns.countplot(y_train)
# plt.subplot(2,1,2)
# g1 = sns.countplot(y_dev)
# plt.show()

# Reshape image in 3 dimensions (height = 28px, width = 28px , canal = 1)
X_train = X_train.values.reshape(-1,28,28,1)
X_dev = X_dev.values.reshape(-1,28,28,1)

# Encode labels to one hot vectors (ex : 2 -> [0,0,1,0,0,0,0,0,0,0])
y_train = to_categorical(y_train, num_classes = 10)
################################################
# Preprocessed data:
# X_train (37800, 784)
# training_label (37800, 1)
# X_dev (4200, 784)
# dev_label (4200,1)
#
# 4. Setup Model
##
################################################


model = Sequential()
model.add(Conv2D(filters=32,
				 kernel_size=(5,5),
				 padding = "Same", 
				 activation='relu',
				 input_shape=(28,28,1)))
model.add(Conv2D(filters=32,
				 kernel_size=(5,5),
				 padding = "Same", 
				 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters=64,
				 kernel_size=(3,3),
				 padding = "Same", 
				 activation='relu'))
model.add(Conv2D(filters=64,
				 kernel_size=(3,3),
				 padding = "Same", 
				 activation='relu'))
model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation = "relu"))
model.add(Dropout(0.5))
model.add(Dense(10, activation = "softmax"))

#Set optimizer
optimizer = RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)
model.compile(optimizer = optimizer , loss = "categorical_crossentropy", metrics=["accuracy"])

epochs = 1 # Turn epochs to 30 to get 0.9967 accuracy
batch_size = 86
history = model.fit(X_train,y_train, 
					batch_size=batch_size,
                              epochs = epochs, 
                              validation_data = (X_dev,y_dev),
                              verbose = 2
                              )









