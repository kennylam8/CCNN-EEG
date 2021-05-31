""" This autoencoder implementation copied and use with minor changes from https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1"""

import torch
import torch.nn as nn
import torch.optim as optim
import scipy.io

batch_size = 512
epochs = 25
learning_rate = 0.001


class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=512
        )
        self.encoder_output_layer = nn.Linear(
            in_features=512, out_features=256
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=256, out_features=512
        )
        self.decoder_output_layer = nn.Linear(
            in_features=512, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.sigmoid(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.sigmoid(activation)
        return reconstructed

    def encode(self, features):
        with torch.no_grad():
            activation = self.encoder_hidden_layer(features)
            activation = torch.relu(activation)
            code = self.encoder_output_layer(activation)
            code = torch.sigmoid(code)
            return code



def train():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AE(input_shape=3072).to(device) #3072
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # load the dataset
    mat = scipy.io.loadmat('alcoholism/uci_eeg_images_v2.mat')
    train_dataset = mat["data"]
    print(train_dataset.shape)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )

    print(train_loader)
    for epoch in range(epochs):
        loss = 0
        for batch_features in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device
            batch_features = batch_features.to(device)
            PRE = batch_features.shape[0]
            batch_features = batch_features.reshape(PRE, -1).float()
            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            outputs = model(batch_features)

            # compute training reconstruction loss
            train_loss = criterion(outputs, batch_features)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, recon loss = {:.8f}".format(epoch + 1, epochs, loss))

    torch.save(model.state_dict(), "./model/model.pth")
    # test_input.to_pickle("./model/test_input.pkl")
    # test_target.to_pickle("./model/test_target.pkl")

# train()  # commented out as we don't want it to update the model everytime this file is called