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

# define a customised neural network structure
class CasPer(torch.nn.Module):

    def __init__(self, n_input, n_output):
        super(CasPer, self).__init__()
        # self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.hidden_list = torch.nn.ModuleList()
        self.out = torch.nn.Linear(n_input, n_output)
        self.iteration = 0
        self.n_output = n_output
        self.last_weight = None
        self.last_bias = None
        self.tempOut = None
        self.new_hidden = None

    def add_neuron(self):
        self.new_hidden = torch.nn.Linear(self.iteration - 1 +
                                                input_neurons, 1)
        self.out = torch.nn.Linear(self.iteration + input_neurons, self.n_output)
        with torch.no_grad():
            for x in range(len(self.out.weight)):
                for ii in range(self.last_weight.size()[1]):
                    self.out.weight[x, ii] = self.last_weight[x, ii]
        self.out.bias = self.last_bias

    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        if iteration == 0:
            return self.out(x)
        if iteration == 1:
            ho = x
            output = x
            ho = self.new_hidden(output)
            ho = torch.tanh(ho)
            output = torch.cat((output, ho), 1)
            y_pred = self.out(output)
            # print(x)
            return y_pred
        else:
            # print("in_feature: ", self.new_hidden.in_features, "iteration: ",self.iteration)
            ho = x
            output = x
            for hidden in self.hidden_list:
                # print(hidden)
                ho = hidden(output)
                ho = torch.sigmoid(ho)
                output = torch.cat((output, ho), 1)
            # print(output.size())
            ho = self.new_hidden(output)
            ho = torch.sigmoid(ho)
            output = torch.cat((output, ho), 1)
            y_pred = self.out(output)
            return y_pred


# define a neural network using the customised structure
net = CasPer(input_neurons, output_neurons)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
# optimiser = torch.optim.Rprop(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []
max_iter = 8
# print(net.parameters())
# train a neural network
for iteration in range(max_iter):
    # optimiser = torch.optim.Rprop(net.parameters(), lr=learning_rate)
    if iteration > 0:
        optimiser = torch.optim.Rprop([
            {"params": net.out.parameters(), "lr": learning_rate*3},
            {"params": net.hidden_list.parameters(), "lr": learning_rate},
            {'params': net.new_hidden.weight, 'lr': learning_rate*6},
            {'params': net.new_hidden.bias, 'lr': learning_rate}
        ])
    else:
        optimiser = torch.optim.Rprop([
            {"params": net.out.parameters(), "lr": learning_rate},
            {"params": net.hidden_list.parameters(), "lr": 0},
        ], lr=learning_rate, )
    for epoch in range(num_epochs):
        # Perform forward pass: compute predicted y by passing x to the model.
        Y_pred = net(X)

        # Compute loss
        loss = loss_func(Y_pred, Y)

        all_losses.append(loss.item())

        # print progress
        if epoch % 50 == 0:
            # convert three-column predicted Y values to one column for comparison
            _, predicted = torch.max(Y_pred, 1)

            # calculate and print accuracy
            total = predicted.size(0)
            correct = predicted.data.numpy() == Y.data.numpy()

            print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
                  % (
                      epoch + 1, num_epochs, loss.item(), 100 * sum(correct) / total))

        # Clear the gradients before running the backward pass.
        net.zero_grad()

        # Perform backward pass
        loss.backward()

        # Calling the step function on an Optimiser makes an update to its
        # parameters
        optimiser.step()
    # print("weight: " , net.out.weight)
    # print("bias: ", net.out.bias)
    # print(iteration,max_iter)
    if iteration == max_iter - 1:
        break
    if net.new_hidden is not None:
        # print(net.new_hidden)
        net.hidden_list.append(net.new_hidden)
        HOW = net.hidden_list[0].weight[0][0]

        print(net.hidden_list[0].weight[0][0] == HOW)

    net.last_weight = net.out.weight
    net.last_bias = net.out.bias
    net.iteration += 1
    net.add_neuron()

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