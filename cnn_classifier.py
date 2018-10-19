#!/usr/local/bin/python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

## Steps:
# 1. Load Data
# 2. Preprocess data
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

raw_training_data = pd.read_csv("./train.csv")
training_set = raw_training_data.sample(frac=0.9)
dev_set = raw_training_data.drop(training_set.index)
del(raw_training_data) # release RAM

training_labels = training_set['label']
training_data = training_set.drop(columns='label')
dev_labels = dev_set['label']
dev_data = dev_set.drop(columns='label')

# normalization
training_data = training_data/255.0
dev_data = dev_data/255.0

del(dev_set,training_set)


################################################
# Preprocessed data:
# training_data (37800, 784)
# training_label (37800, 1)
# dev_data (4200, 784)
# dev_label (4200,1)

# 4. Setup Model
#      &
# 2. Preprocess Data
#      &
# 3. Split into Training&Dev Sets
##
################################################























