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
from draw_subject_for_model_selection import get_ms_subject_ids
from main import main_casper

mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')

# safe the data to dataFrame for easier handling
df = pd.DataFrame.from_dict(mat['data'])
df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
df['y_stimulus_1'] = (df['y_stimulus'] == 1).astype(int)
df['y_stimulus_2'] = (df['y_stimulus'] == 2).astype(int)
df['y_stimulus_3'] = (df['y_stimulus'] == 3).astype(int)
df['y_stimulus_4'] = (df['y_stimulus'] == 4).astype(int)
df['y_stimulus_5'] = (df['y_stimulus'] == 5).astype(int)
df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T
df = df[(df['subjectid'].isin(get_ms_subject_ids()))]
df = df.sample(frac=1)


def ten_fold_cv_within(data):
    tenth = round(len(data)/10)
    total_score = []
    for i in range(10):
        if i <= 9:
            test_data = data[tenth * i :tenth*i + tenth]
            train_data = data[~(data.index.isin(test_data.index))]
        else:
            test_data = data[tenth * i:]
            train_data = data[~(data.index.isin(test_data.index))]

        train_data = train_data.drop(columns=['subjectid','y_stimulus'])
        test_data = test_data.drop(columns=['subjectid','y_stimulus'])
        n_features = train_data.shape[1] - 1
        train_input = train_data.iloc[:, :n_features]
        train_target = train_data.iloc[:, n_features]
        test_input = test_data.iloc[:, :n_features]
        test_target = test_data.iloc[:, n_features]
        score, _ = main_casper(n_features, train_input, train_target, test_input, test_target,
                               0.001, "within_subject", train_data)
        total_score.append(score)
    print(total_score)
    print(sum(total_score) / total_score.__len__())
    return sum(total_score) / total_score.__len__()

def leave_one_subject_out_CV(data):
    tenth = round(len(data)/10)
    total_score = []
    for subj in data['subjectid'].unique():
        test_data = df[df['subjectid'] == subj]
        train_data = df[~(df['subjectid'] == subj)]
        train_data = train_data.drop(columns=['subjectid','y_stimulus'])
        test_data = test_data.drop(columns=['subjectid','y_stimulus'])
        n_features = train_data.shape[1] - 1
        train_input = train_data.iloc[:, :n_features]
        train_target = train_data.iloc[:, n_features]
        test_input = test_data.iloc[:, :n_features]
        test_target = test_data.iloc[:, n_features]
        score, _ = main_casper(n_features, train_input, train_target, test_input, test_target,
                               0.15, "cross_subject", train_data)
        total_score.append(score)
    print(total_score)
    print(sum(total_score) / total_score.__len__())

# uncomment another one if we are tuning cross-subject / within-subject
# ten_fold_cv_within(df)  # within subject
leave_one_subject_out_CV(df)  # cross subject
