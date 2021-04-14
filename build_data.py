import pandas as pd
import torch
import torch.nn.functional as F
import scipy.io


def get_data(train_percent, val_percent, test_percent):
    mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')

    # safe the data to dataFrame for easier handling
    df = pd.DataFrame.from_dict(mat['data'])
    df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
    df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
    df['trialnum'] = pd.DataFrame.from_dict(mat['trialnum']).T
    df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T
    print(df)

    # shuffle data
    df = df.sample(frac=1)
    # print(df)
    # randomly split data into 70/15/15
    train_data_proportion = train_percent
    val_data_proportion = val_percent
    test_data_proportion = test_percent
    sum_amount = train_data_proportion + val_data_proportion + \
                 test_data_proportion

    train_amount = round((train_data_proportion / sum_amount) * len(df))
    val_amount = round((val_data_proportion / sum_amount) * len(df))

    train_data = df[:train_amount]
    val_data = df[train_amount: train_amount + val_amount]
    test_data = df[train_amount + val_amount:]

    # print(train_data)
    # print(val_data)
    # print(test_data)
    print("If this is not 11057 there is something wrong: ",
          len(train_data) + len(val_data) + len(test_data))

    n_features = train_data.shape[1] - 1
    # print(n_features)
    # split training data into input and target
    train_input = train_data.iloc[:, :n_features]
    train_target = train_data.iloc[:, n_features]

    # normalise training data by columns
    # for column in train_input:
    #    train_input[column] = train_input.loc[:, [column]].apply(lambda x:
    #    (x - x.min()) / (x.max()
    #    - x.min()))
    # train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x
    # - x.mean()) / x.std())

    # split training data into input and target
    # the first 9 columns are features, the last one is target
    test_input = test_data.iloc[:, :n_features]
    test_target = test_data.iloc[:, n_features]

    # normalise testing input data by columns
    # for column in test_input:
    #    test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x
    #    - x.min()) / (x.max() -

    # x.min()))
    # test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x -
    # x.mean()) / x.std())
    print(train_input.shape[0])
    print(train_data.shape[0])
    print(train_target.shape[0])

    return n_features, train_input, train_target, test_input, test_target
