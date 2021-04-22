"""
This script provides an example of building a binary neural
network for classifying glass identification dataset on
http://archive.ics.uci.edu/ml/datasets/Glass+Identification
"""

# import libraries
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.io
import matplotlib.pyplot as plt
#print(df)
def twospirals(n_points, noise=.5):
    """
     Returns the two spirals dataset.
    """
    n = np.sqrt(np.random.rand(n_points,1)) * 780 * (2*np.pi)/360
    d1x = -np.cos(n)*n + np.random.rand(n_points,1) * noise
    d1y = np.sin(n)*n + np.random.rand(n_points,1) * noise
    return (np.vstack((np.hstack((d1x,d1y)),np.hstack((-d1x,-d1y)))),
            np.hstack((np.zeros(n_points),np.ones(n_points))))

X, y = twospirals(1000)
plt.title('training set')
plt.plot(X[y==0,0], X[y==0,1], '.', label='class 1')
plt.plot(X[y==1,0], X[y==1,1], '.', label='class 2')
plt.legend()
plt.show()

print(X)
df = pd.DataFrame(X)
df['y'] = y.T
print(df)
df = df.sample(frac=1)
# df = pd.read_csv('alcoholism/spiral.txt', sep=" ")
# print(df)

# randomly split data into 70/15/15
train_data_proportion = 70
val_data_proportion = 0
test_data_proportion = 0
sum_amount = train_data_proportion + val_data_proportion + test_data_proportion

train_amount = round((train_data_proportion / sum_amount) * len(df))
val_amount = round((val_data_proportion / sum_amount) * len(df))

train_data = df[:train_amount]
val_data = df[train_amount: train_amount + val_amount]
test_data = df[train_amount + val_amount:]



# print(train_data)
# print(val_data)
# print(test_data)
print("If this is not 11057 there is something wrong: ", len(train_data)+len(val_data)+len(test_data))

n_features = train_data.shape[1] - 1
# print(n_features)
# split training data into input and target
train_input = train_data.iloc[:, :n_features]
train_target = train_data.iloc[:, n_features]

# normalise training data by columns
# for column in train_input:
#    train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max()
#    - x.min()))
# train_input[column] = train_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# split training data into input and target
# the first 9 columns are features, the last one is target
test_input = test_data.iloc[:, :n_features]
test_target = test_data.iloc[:, n_features]

# normalise testing input data by columns
# for column in test_input:
#    test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.min()) / (x.max() -

# x.min()))
# test_input[column] = test_input.loc[:, [column]].apply(lambda x: (x - x.mean()) / x.std())

# create Tensors to hold inputs and outputs
X = torch.Tensor(train_input.values).float()
Y = torch.Tensor(train_target.values).long()

# define the number of inputs, classes, training epochs, and learning rate
input_neurons = n_features
output_neurons = 2
learning_rate = 0.01
num_epochs = 1
max_iter = 8

# define a neural network using the customised structure

from main import main_casper
net = main_casper(n_features, train_input, train_target, test_input, test_target, learning_rate,num_epochs ,max_iter)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
# optimiser = torch.optim.Rprop(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# Optional: plotting historical loss from ``all_losses`` during network learning
# Please uncomment me from next line to ``plt.show()`` if you want to plot loss

# import matplotlib.pyplot as plt
#
# plt.figure()
# plt.plot(all_losses)
# plt.show()

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every glass (rows)
which class the network guesses (columns).

"""

confusion = torch.zeros(output_neurons, output_neurons)

Y_pred = net(X)

_, predicted = torch.max(Y_pred, 1)

for i in range(train_data.shape[0]):
    actual_class = Y.data[i]
    predicted_class = predicted.data[i]

    confusion[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for training:')
print(confusion)

"""
Step 3: Test the neural network

Pass testing data to the built neural network and get its performance
"""
X_test, Y_test = twospirals(5000)
# create Tensors to hold inputs and outputs
X_test = torch.Tensor(X_test).float()
Y_test = torch.Tensor(Y_test).long()

# test the neural network using testing data
# It is actually performing a forward pass computation of predicted y
# by passing x to the model.
# Here, Y_pred_test contains three columns, where the index of the
# max column indicates the class of the instance
Y_pred_test = net(X_test)

# get prediction
# convert three-column predicted Y values to one column for comparison
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
print('')
print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

"""
Evaluating the Results

To see how well the network performs on different categories, we will
create a confusion matrix, indicating for every iris flower (rows)
which class the network guesses (columns).

"""

confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_data.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('Confusion matrix for testing:')
print(confusion_test)

X_test, Y_test = twospirals(5000)
X_test = torch.Tensor(X_test).float()
Y_test = torch.Tensor(Y_test).long()
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)

plt.title('Neural Network result')
plt.plot(X_test[confusion_test==0,0], X_test[confusion_test==0,1], '.')
plt.plot(X_test[confusion_test==1,0], X_test[confusion_test==1,1], '.')
plt.show()