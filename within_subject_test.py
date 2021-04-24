import torch
import pandas as pd
from main import CasPer

# if you want to return the output of run
from main import main_casper
# model = main_casper.<locals>.CasPer
net = torch.load("model/model.pth")
# net.eval()
test_input = pd.read_pickle("backup/test_input.pkl")
test_target = pd.read_pickle("backup/test_target.pkl")
X_test = torch.Tensor(test_input.values).float()
Y_test = torch.Tensor(test_target.values).long()

confusion = torch.zeros(2, 2)

Y_pred = net(X_test)

# predicted = torch.where(Y_pred < 0.5, 0, 1)
_, predicted = torch.max(Y_pred, 1)
for i in range(X_test.shape[0]):
    actual_class = Y_test.data[i]
    predicted_class = predicted.data[i]
    confusion[int(actual_class)][int(predicted_class)] += 1

print('')
print('Confusion matrix for training:')
print(confusion)


Y_pred_test = net(X_test)
_, predicted_test = torch.max(Y_pred_test, 1)

# calculate accuracy
total_test = predicted_test.size(0)
correct_test = sum(predicted_test.data.numpy() == Y_test.data.numpy())
print('')
print(
    'Validation/Testing Accuracy: %.2f %%' % (100 * correct_test / total_test))

