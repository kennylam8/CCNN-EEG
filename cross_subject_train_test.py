import statistics
import pandas as pd
import scipy.io
import random
import numpy as np
import torch

from draw_subject_for_model_selection import get_test_subject_ids
from casper_main import main_casper


mat = scipy.io.loadmat('alcoholism/uci_eeg_images_v2.mat')
data = mat["data"]
PRE = data.shape[0]
data = data.reshape(PRE, -1)
print(data.shape)

from Autoencoder import AE
model_net = AE(input_shape=3072)
model_net.load_state_dict(torch.load("./autoencoder_model/autoencoder_model.pth"))
model_net.eval()

mol = model_net.encode(torch.tensor(data).float())
mol = np.array(mol)

df = pd.DataFrame(mol)



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
df = df[(df['subjectid'].isin(get_test_subject_ids()))]
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

for i in range(1,12):
    lss = fold_test_set[10 * (i - 1):10 * i]
    df_rem = df[~(df['subjectid'].isin(lss))]
    df_70 = df[(df['subjectid'].isin(lss))]
    train_lst = train_lst+lss

    n_features, train_input, train_target, test_input, test_target, train_data = _10_fold_CV(df, lss)
    model = main_casper(n_features, train_input, train_target, test_input, test_target, 0.08643845, 0.00768044, 0.01173069, 7, 49, "within-subject", train_data)
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
