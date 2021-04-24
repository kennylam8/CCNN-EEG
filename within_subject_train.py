import sys

import pandas as pd
import scipy.io
import random
import numpy as np
import torch

from main import main_casper

mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')

# safe the data to dataFrame for easier handling
df = pd.DataFrame.from_dict(mat['data'])
df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T

# seed = random.randrange((2**32)-1)
# print("Seed was:", seed)

df = df.sample(frac=1)
# df = df.sample(frac=1,random_state=seed)
# random.seed(seed)

learning_rate = 1
num_epochs = 500
max_iter = 6

def train():

    train_data_proportion = 70
    val_data_proportion = 15
    test_data_proportion = 15
    sum_amount = train_data_proportion + val_data_proportion + test_data_proportion

    train_amount = round((train_data_proportion / sum_amount) * len(df))
    val_amount = round((val_data_proportion / sum_amount) * len(df))

    train_data = df[:train_amount]
    val_data = df[train_amount: train_amount + val_amount]
    test_data = df[train_amount + val_amount:]

    print("If this is not 11057 there is something wrong: ",
          len(train_data) + len(val_data) + len(test_data))

    # print("Train data non alcoholic amount: ",
    #       train_data[train_data['y_alcoholic'] == 0][
    #           'subjectid'].unique().__len__())
    # print("Train data alcoholic amount: ",
    #       train_data[train_data['y_alcoholic'] == 1][
    #           'subjectid'].unique().__len__())
    # print("Test data non alcoholic amount: ",
    #       test_data[test_data['y_alcoholic'] == 0][
    #           'subjectid'].unique().__len__())
    # print("Test data alcoholic amount: ",
    #       test_data[test_data['y_alcoholic'] == 1][
    #           'subjectid'].unique().__len__())

    n_features = train_data.shape[1] - 1
    # split training data into input and target
    train_input = train_data.iloc[:, :n_features]
    train_target = train_data.iloc[:, n_features]

    # split training data into input and target
    # the first 9 columns are features, the last one is target
    val_input = val_data.iloc[:, :n_features]
    val_target = val_data.iloc[:, n_features]

    model = main_casper(n_features, train_input, train_target, val_input, val_target,
                        learning_rate, num_epochs, max_iter, 0)

    test_input = test_data.iloc[:, :n_features]
    test_target = test_data.iloc[:, n_features]
    return model, test_input, test_target

# def store_net():
#     global nn_model
#     nn_model = train()
#
# def get_net():
#     return nn_model


model, test_input, test_target = train()
nn_model = model[1]


torch.save(nn_model, "backup/model.pth")
test_input.to_pickle("./model/test_input.pkl")
test_target.to_pickle("./model/test_target.pkl")

