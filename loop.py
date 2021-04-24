import sys

import pandas as pd
import scipy.io
import random
import numpy as np

from main import main_casper

mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')

# safe the data to dataFrame for easier handling
df = pd.DataFrame.from_dict(mat['data'])
df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
df['trialnum'] = pd.DataFrame.from_dict(mat['trialnum']).T
df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T

seed = random.randrange((2**32)-1)
rng = random.Random(seed)
print("Seed was:", seed)

# df = df.sample(frac=1)
df = df.sample(frac=1,random_state=seed)
random.seed(seed)

learning_rate = 1
num_epochs = 500
max_iter = 6

def loop(data_frame, ii, mode, lst):
    # print(np.random.get_state())
    # data_frame = data_frame.sample(frac=1)
    # print("ii",ii)

    if mode == 'leave-one-out':
        train_data = data_frame[~(data_frame['subjectid'] == ii)]
        test_data = data_frame[(data_frame['subjectid'] == ii)]
        # print(train_data)
    elif mode == '10-fold':
        train_data = data_frame[~(data_frame['subjectid'].isin(lst))]
        test_data = data_frame[(data_frame['subjectid'].isin(lst))]

    # print(train_data)
    # a = list(df.subjectid.unique())
    # random.shuffle(a)
    # train_data = df[df['subjectid'].isin(a[:84])]
    # test_data = df[df['subjectid'].isin(a[84:])]

    print("Train data non alcoholic amount: ",
          train_data[train_data['y_alcoholic'] == 0][
              'subjectid'].unique().__len__())
    print("Train data alcoholic amount: ",
          train_data[train_data['y_alcoholic'] == 1][
              'subjectid'].unique().__len__())
    print("Test data non alcoholic amount: ",
          test_data[test_data['y_alcoholic'] == 0][
              'subjectid'].unique().__len__())
    print("Test data alcoholic amount: ",
          test_data[test_data['y_alcoholic'] == 1][
              'subjectid'].unique().__len__())

    train_data = train_data.drop(columns=['subjectid', 'trialnum'])
    test_data = test_data.drop(columns=['subjectid', 'trialnum'])

    n_features = train_data.shape[1] - 1
    # split training data into input and target
    train_input = train_data.iloc[:, :n_features]
    train_target = train_data.iloc[:, n_features]

    # split training data into input and target
    # the first 9 columns are features, the last one is target
    test_input = test_data.iloc[:, :n_features]
    test_target = test_data.iloc[:, n_features]

    # print(train_input)
    # print(test_input)
    return (n_features, train_input, train_target, test_input, test_target)


acc = []
# for i in df.subjectid.unique():
#     n_features, train_input, train_target, test_input, test_target = loop(df, i)
#     # print(n_features, train_input, train_target, test_input, test_target)
#     acc.append(main_casper(n_features, train_input, train_target, test_input, test_target))

aaa = list(df.subjectid.unique())

random.shuffle(aaa)
train_lst = []

# for _ in range(1):
for i in range(1, 11):
    print(i)
    if i == 9:
        lss = aaa[12*(i-1):12*i+1]
    if i == 10:
        lss = aaa[12*(i-1)+1:12*i+1]
    else:
        lss = aaa[12 * (i - 1):12 * i]

    # df_rem = df[df['subjectid'].isin(aaa[84:])]
    # df_70 = df[df['subjectid'].isin(aaa[:84])]

    df_rem = df[~(df['subjectid'].isin(lss))]
    df_70 = df[(df['subjectid'].isin(lss))]
    train_lst = train_lst+lss

    if len(df_rem) + len(df_70) != 11057:
        print(len(df_rem))
        print(len(df_70))
        raise ValueError

    # print(df_70)
    n_features, train_input, train_target, test_input, test_target = loop(df, 123, '10-fold', lss)
    # print(n_features, train_input, train_target, test_input,
    # test_target)
    acc.append(main_casper(n_features, train_input, train_target, test_input, test_target, learning_rate,num_epochs,max_iter, seed))

print(sum(acc) / len(acc))
print(acc)
# print(train_lst)
# print(train_lst)
