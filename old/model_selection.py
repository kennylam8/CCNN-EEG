"""
The Casper algorithm is implemented base on the sample code provided by COMP4660
Lab content, algorithm/codes including normalization, plotting loss diagram and plotting
error matrix and printing epoch/loss and accuracy has been copied directly from the COMP4660
Lab material. Some of which, as denoted in the lab material, is provided by UCI originally serve
as an example for the classfication of the glass dataset
url: http://archive.ics.uci.edu/ml/datasets/Glass+Identification

Other codes however, written entirely on my own (Kenny Lam).
"""

import statistics
import pandas as pd
import scipy.io
import random
import numpy as np
import torch

from draw_subject_for_model_selection import get_ms_subject_ids
from main import main_casper




mat = scipy.io.loadmat('alcoholism/uci_eeg_images_v2.mat')
data = mat["data"]
PRE = data.shape[0]
data = data.reshape(PRE, -1)
print(data.shape)
# df = pd.DataFrame.from_dict(data)
# print(df.shape)

from lab4_task1_cnn_answers import AE
model_net = AE(input_shape=3072)
model_net.load_state_dict(torch.load("./model/old20.pth"))
model_net.eval()

aaa2 = model_net.encode(torch.tensor(data).float())
aaa2 = np.array(aaa2)
# print("ADW", aaa2)
#
# print(aaa2 ,"HEY")

df = pd.DataFrame(aaa2)



# mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')
# df = pd.DataFrame.from_dict(mat['data'])

# safe the data to dataFrame for easier handling
df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
# df['trialnum'] = pd.DataFrame.from_dict(mat['trialnum']).T
df['y_stimulus_1'] = (df['y_stimulus'] == 1).astype(int)
df['y_stimulus_2'] = (df['y_stimulus'] == 2).astype(int)
df['y_stimulus_3'] = (df['y_stimulus'] == 3).astype(int)
df['y_stimulus_4'] = (df['y_stimulus'] == 4).astype(int)
df['y_stimulus_5'] = (df['y_stimulus'] == 5).astype(int)
df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T
df = df[(df['subjectid'].isin(get_ms_subject_ids()))]
df = df.sample(frac=1)

# normalise training data by columns (commented out because not useful)
# for column in df:
#     if (column != "trialnum" and column != "subjectid" and column != "y_alcoholic" and column != "y_stimulus"):
#        df[column] = df.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() - x.min()))
#        df[column] = df.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())
print(df.shape)

def _10_fold_CV(data_frame, lst):
    print(lst)
    train_data = data_frame[~(data_frame['subjectid'].isin(lst))]
    test_data = data_frame[(data_frame['subjectid'].isin(lst))]
    train_data = train_data.drop(columns=['subjectid', 'y_stimulus'])
    test_data = test_data.drop(columns=['subjectid', 'y_stimulus'])


    n_features = train_data.shape[1] - 1
    train_input = train_data.iloc[:, :n_features]
    train_target = train_data.iloc[:, n_features]
    test_input = test_data.iloc[:, :n_features]
    test_target = test_data.iloc[:, n_features]
    return n_features, train_input, train_target, test_input, test_target, train_data


fold_test_set = list(df.subjectid.unique())
random.shuffle(fold_test_set)
train_lst = []
sensitivity = []
accuracy = []
loss = []
num_epoch = []
num_neuron = []
time= []






for i in range(10):
    lss = [fold_test_set[i]]
    df_rem = df[~(df['subjectid'].isin(lss))]
    df_70 = df[(df['subjectid'].isin(lss))]
    train_lst = train_lst+lss

    n_features, train_input, train_target, test_input, test_target, train_data = _10_fold_CV(df, lss)
    model = main_casper(n_features, train_input, train_target, test_input, test_target, 0.15, 0.0005, 0.01, 10, 50, "within-subject", train_data)
    accuracy.append(float(model[0]))
    sensitivity.append(float(model[1]))
    loss.append(float(model[3]))
    num_epoch.append(float(model[2]))
    num_neuron.append(float(model[4]))
    time.append(float(model[5]))


for x in range(6):
    name = ['sensitivity','accuracy','loss','num_epoch','num_neuron','time'][x]
    lst = [sensitivity, accuracy, loss, num_epoch, num_neuron, time][x]
    print(name, " mean: ", statistics.mean(lst))
    print(name, " median: ", statistics.median(lst))
    print(name, " sd: ", statistics.stdev(lst))


