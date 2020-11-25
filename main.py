import argparse
from time import time

import numpy as np
import numpy.random as nr
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model import MultiLayerPerceptron
from dataset import AdultDataset
from util import *

from scipy.signal import savgol_filter


""" Adult income classification

In this lab we will build our own neural network pipeline to do classification on the adult income dataset. More
information on the dataset can be found here: http://www.cs.toronto.edu/~delve/data/adult/adultDetail.html

"""
seed = 0

# =================================== LOAD DATASET =========================================== #

######

data = pd.read_csv('adult.csv')

######

# =================================== DATA VISUALIZATION =========================================== #

# the dataset is imported as a DataFrame object, for more information refer to
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# we can check the number of rows and columns in the dataset using the .shape field
# to get a taste of what our datset looks like, let's visualize the first 5 rows of the dataset using the .head() method
# the task will be to predict the "income" field (>50k or <50k) based on the other fields in the dataset
# check how balanced our dataset is using the .value_counts() method.

######

print(data.shape)
print(data.columns)
print(data.head())
print(data["income"].value_counts())

######


# =================================== DATA CLEANING =========================================== #

# datasets often come with missing or null values, this is an inherent limit of the data collecting process
# before we run any algorithm, we should clean the data of any missing values or unwanted outliers which could be
# detrimental to the performance or training of the algorithm. In this case, we are told that missing values are
# indicated with the symbol "?" in the dataset

# let's first count how many missing entries there are for each feature
col_names = data.columns
num_rows = data.shape[0]
for feature in col_names:
    pass
    ######
    missing = data[feature].isin(["?"]).sum()
    print(feature + " missing entries: " + str(missing))
    ######

# next let's throw out all rows (samples) with 1 or more "?"
# Hint: take a look at what data[data["income"] != ">50K"] returns
# Hint: if data[field] do not contain strings then data["income"] != ">50K" will return an error

    ######

    data = data[data[feature] != "?"]

    ######

print(data.shape)

# =================================== BALANCE DATASET =========================================== #

    ######

data_under = data[data["income"] == "<=50K"]
data_over = data[data["income"] == ">50K"]

data_under = data_under.sample(n=11208, random_state=1)
data = pd.concat([data_under, data_over])

print(data["income"].value_counts())


    ######

# =================================== DATA STATISTICS =========================================== #

# our dataset contains both continuous and categorical features. In order to understand our continuous features better,
# we can compute the distribution statistics (e.g. mean, variance) of the features using the .describe() method

######

print(data.describe())

######

# likewise, let's try to understand the distribution of values for discrete features. More specifically, we can check
# each possible value of a categorical feature and how often it occurs
categorical_feats = ['workclass', 'race', 'education', 'marital-status', 'occupation',
                    'relationship', 'gender', 'native-country', 'income']

for feature in categorical_feats:
    pass
    ######

    print(feature + ": ")
    print(data[feature].value_counts())

    ######

# visualize the first 3 features using pie and bar graphs

######

# for i in categorical_feats:
#     pie_chart(data, i)
#     binary_bar_chart(data, i)

######

# =================================== DATA PREPROCESSING =========================================== #

# we need to represent our categorical features as 1-hot encodings
# we begin by converting the string values into integers using the LabelEncoder class
# next we convert the integer representations into 1-hot encodings using the OneHotEncoder class
# we don't want to convert 'income' into 1-hot so let's extract this field first
# we also need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# don't forget to stitch continuous and cat features together

# ENCODE CATEGORICAL FEATURES

# Helpful Hint: .values converts the DataFrame to a numpy array

# LabelEncoder information: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.LabelEncoder.html
#       the LabelEncoder works by transforming the values in an input feature into a 0-to-"n_classes-1" digit label 
#       if a feature in the data has string values "A, B, X, Y", then LabelEncoder will turn these into the numeric 0, 1, 2, 3
#       like other scikit-learn objects, the LabelEncoder must first fit() on the target feature data (does not return anything)
#       fitting on the target feature data creates the mapping between the string values and the numerical labels
#       after fitting, then transform() on a set of target feature data will return the numerical labels representing that data
#       the combined fit_transform() does this all in one step. Check the examples in the doc link above!

label_encoder = LabelEncoder()
######

labelencoder = LabelEncoder()
# -------CODE FOR ENCODER AND ONE HOT ENCODING IN THE NEXT SECTION IN ONE FUNCTION---------

######

# OneHotEncoder information: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
#    the OneHotEncoder works basically identical to the LabelEncoder
#    however, its input, instead of a single numeric array, is a matrix (dense or sparse) of 0 and 1 values
#    consider the following tabular data X of N data points (assume it is a data frame):
#
#    F1     F2      F3
#    ON     1     Toronto
#    ON     3     Scarborough
#    OFF    2     North York
#    ON     3     Toronto
#    OFF    3     Etobicoke
#    OFF    1     Scarborough
#    ...
#
#    F1 has 2 string values (ON, OFF), F2 has 2 numeric values (1, 2, 3), and F3 has 4 string values (Toronto, Scarborough, 
#       North York, Etobicoke)
#    When we use the OneHotEncoder's fit_transform on this data frame X, the resulting matrix takes the shape: N x (2 + 3 + 4)
#
#    [[1 0 1 0 0 1 0 0 0]
#     [1 0 0 0 1 0 1 0 0]
#     [0 1 0 1 0 0 0 1 0]
#     [1 0 0 0 1 1 0 0 0]
#     [0 1 0 0 1 0 0 0 1]
#     [0 1 1 0 0 0 1 0 0]
#    ...
#
#    In other words, for tabular data with N data points and k features F1 ... Fk,
#    Then the resulting output matrix will be of size (N x (F1_n + ... + Fk_n))
#    This is because, looking at datapoint 2 for example: [1 0 0 0 1 0 1 0 0],
#    [1 0 | 0 0 1 | 0 1 0 0] -> here, [1 0] is the encoding for "ON" (ON vs OFF), [0 0 1] is the encoding for "3" (1 vs 2 vs 3), etc.
#    If a single _categorical variable_ F has values 0 ... N-1, then its 1-of-K encoding will be a vector of length F_n
#    where all entries are 0 except the value the data point takes for F at that point, which is 1.
#    Thus, for features F1 ... Fk, as described above, the length-Fi_n encodings are appended horizontally.

# firstly, we need to drop 'income' becaue we don't want to convert it into one-hot encoding:
y = data['income']
data = data.drop(columns=['income'])
categorical_feats.remove('income')
y = y.values  # convert DataFrame to numpy array

y_temp = np.zeros((22416, 1))

for i in range(0,len(y),1): # converts the label to an integer 1 or 0
    if y[i] == '>50K':
        y_temp[i,0] = float(1)
    else:
        y_temp[i,0] = float(0)

y = y_temp


# now, we can use the OneHotEncoder on the part of the data frame encompassed by 'categorial_feats'
# we can fit and transform as usual. Your final output one-hot matrix should be in the variable 'cat_onehot'
oneh_encoder = OneHotEncoder()
######

cat_onehot = np.zeros((22416,1)) # replace this with the output of your fit and transform

def encode_string(cat_feature):
    # encode the strings to numeric categories
    labelencoder.fit(cat_feature)
    enc_cat_feature = labelencoder.transform(cat_feature)
    # apply the one hot encoding
    encoded = oneh_encoder.fit(enc_cat_feature.reshape(-1, 1))
    return encoded.transform(enc_cat_feature.reshape(-1, 1)).toarray()

for col in categorical_feats:
    temp = encode_string(data[col])
    cat_onehot = np.concatenate([cat_onehot, temp], axis = 1)

cat_onehot = np.delete(cat_onehot, 0, 1) # removes the first column of this array which is just zeros

print(cat_onehot.shape)

######


# NORMALIZE CONTINUOUS FEATURES

# finally, we need to preprocess the continuous features by normalizing against the feature mean and standard deviation
# we begin by storing the data dropped of the categorical_feats in a separate variable, 'cts_data'
# your task is to use .mean() and .std() on this data to normalize it, then covert it into a numpy array

cts_data_pd = data.drop(columns=categorical_feats)
cts_data = np.zeros((22416,1))
######

for col in cts_data_pd.columns:
    mean = cts_data_pd[col].mean()
    std = cts_data_pd[col].std()
    data = cts_data_pd[col].values
    data = (data-mean)/std
    data = data.reshape(-1,1)
    cts_data = np.concatenate([cts_data, data], axis = 1)

cts_data = np.delete(cts_data, 0, 1) # removes the first column of this array which is just zeros

######

# finally, we stitch continuous and categorical features
X = np.concatenate([cts_data, cat_onehot], axis=1)
print("Shape of X =", X.shape)

# =================================== MAKE THE TRAIN AND VAL SPLIT =========================================== #
# we'll make use of the train_test_split method to randomly divide our dataset into two portions
# control the relative sizes of the two splits using the test_size parameter

######

random = nr.seed(1000)
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=random) # splits the dataset with a validation size of 20%

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

y_train = y_train.astype(np.float32) # converts to a float to create the tensor

######

# =================================== LOAD DATA AND MODEL =========================================== #

def load_data(batch_size):
    ######

    params = {'batch_size': batch_size, 'shuffle': True}
    train_set = AdultDataset(X_train, y_train)
    train_loader = DataLoader(train_set, **params) # instantiates DataLoader using the given parameters and AdultDataset class

    val_set = AdultDataset(X_test, y_test)
    val_loader = DataLoader(val_set, **params) # similar to above

    ######


    return train_loader, val_loader


def load_model(lr):

    ######

    model = MultiLayerPerceptron(103)
    loss_fnc = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr)

    ######

    return model, loss_fnc, optimizer


def evaluate(model, val_loader):
    total_corr = 0

    ######

    for i in range(0, len(val_loader.dataset), 1):
        data, label = val_loader.dataset.__getitem__(i)
        data = torch.from_numpy(data).float() # gets the data
        label = torch.from_numpy(label).float() # gets the associated labels
        output = model(data) # passes the data through the model that has been trained
        if (float(output.item()) > 0.5 and float(label.item()) == 1) or (float(output.item()) <= 0.5 and float(label.item()) == 0):
            total_corr = total_corr + 1 # determines the total number of correct predictions


    ######

    return float(total_corr)/len(val_loader.dataset) # returns the accuracy


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default = 64)
    parser.add_argument('--lr', type=float, default = 0.15)
    parser.add_argument('--epochs', type=int, default= 8)
    parser.add_argument('--eval_every', type=int, default=20)

    args = parser.parse_args()

    ######

    train, val = load_data(args.batch_size) # loads training and validation data
    model, loss, opt = load_model(args.lr) # instantiates the model, loss function, and optimization function

    valid_acc = [] # array to hold validation accuracies through the epochs
    train_acc = [] # array to hold training accuracies through the epochs
    time_total = [] # array to hold the time increments between steps
    first_time = time() # starts the timer

    for epoch in range(0,args.epochs,1):
        total_loss = 0
        total_corr = 0
        for i, data in enumerate(train, 0):
            inputs, labels = data
            inputs = inputs.float() # inputs of the training data
            labels = labels.float() # labels of the training data

            opt.zero_grad() # sets the gradient to 0

            outputs = model(inputs) # predicts labels based on the inputs

            loss_in = loss(input=outputs, target=labels) # determines the loss
            loss_in.backward()
            opt.step()

            total_loss = total_loss + loss_in.item() # total running loss

            for j in range(0, len(outputs), 1):
                if (float(outputs[j].item()) > 0.5 and float(labels[j].item()) == 1) or (float(outputs[j].item()) <= 0.5 and float(labels[j].item()) == 0):
                    total_corr = total_corr + 1 # determines the number of correct from the training data

            if i % args.eval_every == (args.eval_every - 1):
                train_accuracy = float(total_corr)/(args.eval_every * args.batch_size) # training accuracy
                print("   " + str(total_corr) + " out of " + str(args.eval_every * args.batch_size) + " correct")
                total_corr = 0 # resets the number of correct training predictions
                accuracy = evaluate(model, val) # determines the validation accuracy
                print("   Validation Accuracy: " + str(accuracy))
                valid_acc.append(accuracy)
                train_acc.append(train_accuracy)

                time_diff = time() - first_time # determines the time increment
                print("   Elapsed Time: " + str(time_diff) + " seconds")

                time_total.append(time_diff)
                print('[%d, %5d] loss: %.5f' %
                  (epoch + 1, i + 1, total_loss / args.eval_every)) # prints the epoch and step
                total_loss = 0.0

    ######

    train_acc = savgol_filter(train_acc,5,2) # smooths the training accuracy

    plt.plot(valid_acc, label = "Validation") # plot of training and validation accuracy
    plt.plot(train_acc, label = "Training")
    plt.legend()
    plt.xlabel("Number of Steps")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs. Number of Steps")
    plt.show()

if __name__ == "__main__":
    main()