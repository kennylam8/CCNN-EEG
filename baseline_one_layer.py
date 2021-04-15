# import libraries


import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import scipy.io
from build_data_70_sub import get_data_70

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load data and store it in dictionary

train = 70
validate = 0
test = 30
n_features, train_input, train_target, test_input, test_target = get_data_70(
    train, validate, test)  # refer to build_data.py the

# create Tensors to hold inputs and outputs
X = torch.Tensor(train_input.values).float()
Y = torch.Tensor(train_target.values).long()



# define the number of inputs, classes, training epochs, and learning rate
input_neurons = n_features
hidden_neurons = 10
output_neurons = 2
learning_rate = 0.01
num_epochs = 5000

# define a customised neural network structure
class TwoLayerNet(torch.nn.Module):

    def __init__(self, n_input, n_hidden, n_output):
        super(TwoLayerNet, self).__init__()
        # define linear hidden layer output
        self.hidden = torch.nn.Linear(n_input, n_hidden)
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
        self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)
        # define linear output layer output
        self.out = torch.nn.Linear(n_hidden, n_output)

    def forward(self, x):
        """
            In the forward function we define the process of performing
            forward pass, that is to accept a Variable of input
            data, x, and return a Variable of output data, y_pred.
        """
        h_input1 = self.hidden(x)
        h_output1 = torch.sigmoid(h_input1)
        h_input2 = self.hidden2(h_output1)
        h_output2 = torch.tanh(h_input2)
        # h_input3 = self.hidden2(h_output2)
        # h_output3 = torch.tanh(h_input3)
        # h_input4 = self.hidden2(h_output3)
        # h_output4 = torch.tanh(h_input4)
        # h_input5 = self.hidden2(h_output4)
        # h_output5 = torch.tanh(h_input5)
        y_pred = self.out(h_output2)

        return y_pred


# define a neural network using the customised structure
net = TwoLayerNet(input_neurons, hidden_neurons, output_neurons)

# define loss function
loss_func = torch.nn.CrossEntropyLoss()

# define optimiser
optimiser = torch.optim.Adam(net.parameters(), lr=learning_rate)

# store all losses for visualisation
all_losses = []

# train a neural network
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

        X_test = torch.Tensor(test_input.values).float()
        Y_test = torch.Tensor(test_target.values).long()

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

        print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

    # Clear the gradients before running the backward pass.
    net.zero_grad()

    # Perform backward pass
    loss.backward()

    # Calling the step function on an Optimiser makes an update to its
    # parameters
    optimiser.step()

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

for i in range(train_input.shape[0]):
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

# create Tensors to hold inputs and outputs
def print_test():
    X_test = torch.Tensor(test_input.values).float()
    Y_test = torch.Tensor(test_target.values).long()

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

    print('Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

    """
    Evaluating the Results
    
    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every iris flower (rows)
    which class the network guesses (columns).
    
    """


X_test = torch.Tensor(test_input.values).float()
Y_test = torch.Tensor(test_target.values).long()
Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)
confusion_test = torch.zeros(output_neurons, output_neurons)

for i in range(test_input.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted_test.data[i]

    confusion_test[actual_class][predicted_class] += 1

print('')
print('Confusion matrix for testing:')
print(confusion_test)

# print(net.out.weight)
