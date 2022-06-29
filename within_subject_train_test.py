import statistics
import pandas as pd
import scipy.io
from draw_subject_for_model_selection import get_test_subject_ids
from casper_main import main_casper


mat = scipy.io.loadmat('alcoholism/uci_eeg_features.mat')  # FIXME dataset not avalible in this repo
data = mat["data"]
PRE = data.shape[0]
data = data.reshape(PRE, -1)
print(data.shape)
df = pd.DataFrame.from_dict(data)

# Cast the data to dataFrame for easier handling
df['y_stimulus'] = pd.DataFrame.from_dict(mat['y_stimulus']).T
df['subjectid'] = pd.DataFrame.from_dict(mat['subjectid']).T
df['y_stimulus_1'] = (df['y_stimulus'] == 1).astype(int)
df['y_stimulus_2'] = (df['y_stimulus'] == 2).astype(int)
df['y_stimulus_3'] = (df['y_stimulus'] == 3).astype(int)
df['y_stimulus_4'] = (df['y_stimulus'] == 4).astype(int)
df['y_stimulus_5'] = (df['y_stimulus'] == 5).astype(int)
df['y_alcoholic'] = pd.DataFrame.from_dict(mat['y_alcoholic']).T
df = df[(df['subjectid'].isin(get_test_subject_ids()))]
df = df.sample(frac=1)

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


train_lst = []
sensitivity = []
accuracy = []
loss = []
num_epoch = []
num_neuron = []
time= []

def ccs3 ():
    df_data = df.copy()
    df_data = df_data.sample(frac=1)
    tenth = round(len(df_data) / 10)
    for i in range(10):
        if i <= 9:
            test_data = df_data[tenth * i:tenth * i + tenth]
            train_data = df_data[~(df_data.index.isin(test_data.index))]
        else:
            test_data = df_data[tenth * i:]
            train_data = df_data[~(df_data.index.isin(test_data.index))]
        train_data = train_data.drop(columns=['subjectid', 'y_stimulus'])
        test_data = test_data.drop(columns=['subjectid', 'y_stimulus'])
        n_features = train_data.shape[1] - 1
        train_input = train_data.iloc[:, :n_features]
        train_target = train_data.iloc[:, n_features]
        test_input = test_data.iloc[:, :n_features]
        test_target = test_data.iloc[:, n_features]
        model = main_casper(n_features, train_input, train_target, test_input, test_target, 0.0530317, 0.00388587,
                            0.0143323, 14, 55, "within-subject",
                            train_data)  # 0.08643845, 0.00768044, 0.01173069, 14, 49
        accuracy.append(float(model[0]))
        sensitivity.append(float(model[1]))
        loss.append(float(model[3]))
        num_epoch.append(float(model[2]))
        num_neuron.append(float(model[4]))
        time.append(float(model[5]))


ccs3()
for x in range(6):
    name = ['sensitivity', 'accuracy', 'loss', 'num_epoch', 'num_neuron', 'time'][x]
    lst = [sensitivity, accuracy, loss, num_epoch, num_neuron, time][x]
    print(name, " mean: ", statistics.mean(lst))
    print(name, " median: ", statistics.median(lst))
    print(name, " sd: ", statistics.stdev(lst))
