import pandas as pd
import scipy.io
import random
import matplotlib.pyplot as plt


def get_data_70(train_percent, val_percent, test_percent):
    # def get_data(train_percent, val_percent, test_percent):
    mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')

    # safe the data to dataFrame for easier handling
    df = pd.DataFrame.from_dict(mat['data'])
    df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
    df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
    df['trialnum'] = pd.DataFrame.from_dict(mat['trialnum']).T
    df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T

    a = list(df.subjectid.unique())
    random.shuffle(a)

    df_70 = df[df['subjectid'].isin(a[:84])]
    df_rem = df[df['subjectid'].isin(a[84:])]

    if len(df_rem) + len(df_70) != 11057:
        raise ValueError

    # train_data_proportion = train_percent
    # val_data_proportion = val_percent
    # test_data_proportion = test_percent
    # sum_amount = train_data_proportion + val_data_proportion + \
    #               test_data_proportion
    #
    # train_amount = round((train_data_proportion / sum_amount) * len(df))
    # val_amount = round((val_data_proportion / sum_amount) * len(df))
    #
    # train_data = df[:train_amount]
    # val_data = df[train_amount: train_amount + val_amount]
    # test_data = df[train_amount + val_amount:]

    df = df.sample(frac=1)
    fig = plt.figure()

    # dff = pd.DataFrame()
    # for b in range(1,65):
    #     dff['alpha' + str(b)] = ((df.iloc[:, (b-1)*2: b*2]).sum(axis = 1) / 2)
    # dff['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
    # dff['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
    # dff['trialnum'] = pd.DataFrame.from_dict(mat['trialnum']).T
    # dff['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T
    train_data = df[df['subjectid'].isin(a[:84])]
    test_data = df[df['subjectid'].isin(a[84:])]

    print("Train data non alcoholic amount: ", train_data[train_data['y_alcoholic'] == 0]['subjectid'].unique().__len__())
    print("Train data alcoholic amount: ", train_data[train_data['y_alcoholic'] == 1]['subjectid'].unique().__len__())
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
    return n_features, train_input, train_target, test_input, test_target
