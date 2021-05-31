"""
The Casper algorithm is implemented base on the sample code provided by COMP4660
Lab content, algorithm/codes including normalization, plotting loss diagram and plotting
error matrix and printing epoch/loss and accuracy has been copied directly from the COMP4660
Lab material. Some of which, as denoted in the lab material, is provided by UCI originally serve
as an example for the classfication of the glass dataset
url: http://archive.ics.uci.edu/ml/datasets/Glass+Identification

Other codes however, written entirely on my own (Kenny Lam).
"""
"""
main.py is used for specifying Casper neural network and for training the network, it does not run on its own as it requires dataset as input,
the entire project is done by these steps using these files:
1. draw_subject_for_model_selection.py  for randomly drawing 12 subjects for model selection (redrawn if the 12 subject is extremely unbalanced)
2. perform model selection using model_selection.py and hyper-parameter tuning using those 12 subjects (we used leave-one-subject-out for tuning)
3. run within_subject_train_test.py for within subject testing
4. run cross_subject_train-test.py for cross subject testing
"""
# import libraries
import torch
import time

# define the CasPer network structure
class CasPer(torch.nn.Module):
    def __init__(self, n_input, n_output):
        super(CasPer, self).__init__()
        self.hidden_list = torch.nn.ModuleList()
        self.out = torch.nn.Linear(n_input, n_output)
        self.n_output = n_output
        self.iteration = 0
        self.last_weight = None
        self.last_bias = None
        self.tempOut = None
        self.new_hidden = None
        self.n_input = n_input

    def add_neuron(self):
        self.new_hidden = torch.nn.Linear(self.iteration - 1 + self.n_input,
                                          1)
        self.out = torch.nn.Linear(self.iteration + self.n_input,
                                   self.n_output)
        with torch.no_grad():
            for x in range(len(self.out.weight)):
                for ii in range(self.last_weight.size()[1]):
                    self.out.weight[x, ii] = self.last_weight[x, ii]
        self.out.bias = self.last_bias

    def forward(self, x):
        if self.iteration == 0:
            return self.out(x)
        if self.iteration == 1:
            ho = x
            output = x
            ho = self.new_hidden(output)
            ho = torch.sigmoid(ho)
            output = torch.cat((output, ho), 1)
            y_pred = self.out(output)
            return y_pred
        else:
            output = x
            for hidden in self.hidden_list:
                ho = hidden(output)
                ho = torch.sigmoid(ho)
                output = torch.cat((output, ho), 1)
            ho = self.new_hidden(output)
            ho = torch.sigmoid(ho)
            output = torch.cat((output, ho), 1)
            y_pred = self.out(output)
            return y_pred





def main_casper(n_features, train_input, train_target, test_input, test_target, cutoff, k,checkpoint_min_loss, max_iter,constant_epoch,mode, train_data):

    # define the number of inputs, classes, training epochs, and learning rate
    input_neurons = n_features
    output_neurons = 2
    num_epochs = 5000  # This total number of epoch would not exceed 5000 so this is just for looping
    # max_iter = 6 if mode == "cross-subject" else 15 # maximum of neuron
    t_start = time.time()

    # define a neural network using the customised structure
    net = CasPer(input_neurons, output_neurons)

    # define loss function
    loss_func = torch.nn.CrossEntropyLoss()

    all_losses = []
    all_epoch = 0
    for iteration in range(max_iter):
        net.iteration = iteration

        # Reset the optimiser everytime a new neuron is added
        if net.iteration > 0:
            optimiser = torch.optim.Rprop([
                {"params": net.out.parameters(), "lr": 0.01},
                {"params": net.hidden_list.parameters(), "lr": 0.005},
                {'params': net.new_hidden.weight, 'lr': 0.2},
                {'params': net.new_hidden.bias, 'lr': 0.005}
            ])
        else:
            optimiser = torch.optim.Rprop([
                {"params": net.out.parameters(), "lr": 0.005},
                {"params": net.hidden_list.parameters(), "lr": 0.005},
            ], lr=0.01,)

        P = 50
        # constant_epoch = 50 if mode == "cross-subject" else 500
        checkpoint_epoch = constant_epoch + P * (iteration + 1)
        FLAG = False
        for epoch in range(num_epochs):
            mb = 0

            td = torch.utils.data.DataLoader(dataset=train_data.to_numpy(), batch_size=1024, shuffle=True) \
                if mode == "cross-subject" else [0]

            for batch in td:
                if mode == "cross-subject":
                    X = batch[:,:-1]
                    Y = batch[:,-1]
                    X = (X).float()
                    Y = (Y).long()
                else:
                    X = torch.Tensor(train_input.values).float()
                    Y = torch.Tensor(train_target.values).long()

                all_epoch += 1
                mb += 1
                # Perform forward pass: compute predicted y by passing x to the model.
                Y_pred = net(X)

                # Compute loss

                loss = loss_func(Y_pred, Y)

                # Clear the gradients before running the backward pass.
                net.zero_grad()

                # Perform backward pass
                loss.backward()

                for p in net.parameters():
                    # k = 0.0005
                    p.grad -= k * torch.sign(p.grad) * (p.grad ** 2) * 2 **(-0.01* epoch)

                # Calling the step function on an Optimiser makes an update to its parameters
                optimiser.step()

            if epoch == 0:
                loss_0 = loss.item()
            elif loss.item() < cutoff:
                break

            if epoch == checkpoint_epoch:
                if loss_0 - loss.item() < checkpoint_min_loss:
                    FLAG = True
                    break
                else:
                    break

            # print progress
            if epoch % 50 == 0 and 1==2:
                _, predicted = torch.max(Y_pred, 1)

                # calculate and print accuracy
                total = predicted.size(0)
                correct = predicted.data.numpy() == Y.data.numpy()

                print('Epoch [%d/%d] Loss: %.4f  Accuracy: %.2f %%'
                      % (
                          epoch + 1, checkpoint_epoch, loss.item(),
                          100 * sum(correct) / total))
                all_losses.append(loss.item())


        if loss.item() < cutoff or FLAG:
            break

        if net.new_hidden is not None:
            net.hidden_list.append(net.new_hidden)
            HOW = net.hidden_list[0].weight[0][0]

            print("Number of neuron: ",net.hidden_list.__len__() + 1)

        net.last_weight = net.out.weight
        net.last_bias = net.out.bias
        net.iteration += 1
        net.add_neuron()

    t_end = time.time()
    # Optional: plotting historical loss from ``all_losses`` during network
    # learning
    # Please uncomment me from next line to ``plt.show()`` if you want to plot loss
    # import matplotlib.pyplot as plt
    # plt.figure()
    # plt.plot(all_losses, c='blue', label="Training")
    # plt.xlabel("epoch")
    # plt.ylabel("loss")
    # plt.legend(loc="lower left")
    # plt.show()

    """
    Evaluating the Results
    
    To see how well the network performs on different categories, we will
    create a confusion matrix, indicating for every glass (rows)
    which class the network guesses (columns).
    
    """

    X = torch.Tensor(train_input.values).float()
    Y = torch.Tensor(train_target.values).long()

    print(all_epoch, "all epoch")
    confusion = torch.zeros(2, 2)

    Y_pred = net(X)

    # predicted = torch.where(Y_pred < 0.5, 0, 1)
    _, predicted = torch.max(Y_pred, 1)
    for i in range(train_input.shape[0]):
        actual_class = Y.data[i]
        predicted_class = predicted.data[i]
        confusion[int(actual_class)][int(predicted_class)] += 1

    print('')
    print('Confusion matrix for training:')
    print(confusion)

    """
    Step 3: Test the neural network
    
    Pass testing data to the built neural network and get its performance
    """

    # create Tensors to hold inputs and outputs
    X_test = torch.Tensor(test_input.values).float()
    Y_test = torch.Tensor(test_target.values).long()
    confusion = torch.zeros(2, 2)
    Y_pred_test = net(X_test)

    # get prediction
    # convert three-column predicted Y values to one column for comparison
    _, predicted_test = torch.max(Y_pred_test, 1)

    for i in range(test_input.shape[0]):
        actual_class = Y_test.data[i]
        predicted_class = predicted_test.data[i]

        confusion[int(actual_class)][int(predicted_class)] += 1
    print('')
    print('Confusion matrix for Validation/Testing:')
    print(confusion)
    print("Sensitivity: ", confusion[0][0] / (confusion[0][0] + confusion[1][0]))
    # calculate accuracy
    total_test = predicted_test.size(0)
    correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
    print('')
    print('Validation/Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

    return 100 * correct_test / total_test, confusion[0][0] / (confusion[0][0] + confusion[1][0]), all_epoch, loss.item(), net.iteration, t_end-t_start

