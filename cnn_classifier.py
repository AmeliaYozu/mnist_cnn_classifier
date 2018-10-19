#!/usr/local/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from keras.models import Sequential
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
training_set = raw_data.sample(frac=0.9)
dev_set = raw_data.drop(training_set.index)

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
plt.figure()
plt.subplot(2,1,1)
g = sns.countplot(y_train)
plt.subplot(2,1,2)
g1 = sns.countplot(y_dev)
plt.show()

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
				 padding = "same", 
				 activation='relu',
				 input_shape=(28,28,1)))















