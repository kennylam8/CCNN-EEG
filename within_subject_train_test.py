"""
The Casper algorithm is implemented base on the sample code provided by COMP4660
Lab content, algorithm/codes including normalization, plotting loss diagram and plotting
error matrix and printing epoch/loss and accuracy has been copied directly from the COMP4660
Lab material. Some of which, as denoted in the lab material, is provided by UCI originally serve
as an example for the classfication of the glass dataset
url: http://archive.ics.uci.edu/ml/datasets/Glass+Identification

Other codes however, written entirely on my own (Kenny Lam).
"""

import pandas as pd
import scipy.io
from draw_subject_for_model_selection import get_test_subject_ids
from main import main_casper
import torch
import statistics

# mat = scipy.io.loadmat('alcoholism/uci_eeg_images_v2.mat')
# data = mat["data"]
# PRE = data.shape[0]
# data = data.reshape(PRE, -1)
# print(data.shape)
# safe the data to dataFrame for easier handling
# df = pd.DataFrame.from_dict(data[:,:1500])

mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')
df = pd.DataFrame.from_dict(mat['data'])

print(df)
df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
df['y_stimulus_1'] = (df['y_stimulus'] == 1).astype(int)
df['y_stimulus_2'] = (df['y_stimulus'] == 2).astype(int)
df['y_stimulus_3'] = (df['y_stimulus'] == 3).astype(int)
df['y_stimulus_4'] = (df['y_stimulus'] == 4).astype(int)
df['y_stimulus_5'] = (df['y_stimulus'] == 5).astype(int)
df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T
df = df[(df['subjectid'].isin(get_test_subject_ids()))]  # get subjects that are not used in the hyper-parameter tuning process
df = df.sample(frac=1)
print(df)

def train():

    train_data_proportion = 7
    val_data_proportion = 3
    test_data_proportion = 0
    sum_amount = train_data_proportion + val_data_proportion + test_data_proportion

    train_amount = round((train_data_proportion / sum_amount) * len(df))
    val_amount = round((val_data_proportion / sum_amount) * len(df))

    train_data = df[:train_amount]
    val_data = df[train_amount: train_amount + val_amount]

    train_data = train_data.drop(columns=['subjectid','y_stimulus','y_stimulus'])
    val_data = val_data.drop(columns=['subjectid','y_stimulus','y_stimulus'])

    n_features = train_data.shape[1] - 1
    # split training data into input and target
    train_input = train_data.iloc[:, :n_features]
    train_target = train_data.iloc[:, n_features]

    # split training data into input and target
    # the first 9 columns are features, the last one is target
    val_input = val_data.iloc[:, :n_features]
    val_target = val_data.iloc[:, n_features]
    model = main_casper(n_features, train_input, train_target, val_input, val_target, 0.12, "within_subject", train_data)

    return model


train()
# This part is for generating the testing result for the 50 trail table (Table 1)
sensitvity = []
accuracy = []
loss = []
num_epoch = []
num_neuron = []
time= []
for i in range(10):
    df = df.sample(frac=1)
    model = train()
    accuracy.append(float(model[0]))
    sensitvity.append(float(model[1]))
    loss.append(float(model[3]))
    num_epoch.append(float(model[2]))
    num_neuron.append(float(model[4]))
    time.append(float(model[5]))

for x in range(6):
    name = ['sensitvity','accuracy','loss','num_epoch','num_neuron','time'][x]
    lst = [sensitvity,accuracy,loss,num_epoch,num_neuron,time][x]
    print(name, " mean: ", statistics.mean(lst))
    print(name, " median: ", statistics.median(lst))
    print(name, " sd: ", statistics.stdev(lst))

# torch.save(nn_model, "./model/model.pth")
# test_input.to_pickle("./model/test_input.pkl")
# test_target.to_pickle("./model/test_target.pkl")
