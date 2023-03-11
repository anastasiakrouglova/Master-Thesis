# https://www.youtube.com/watch?v=4p0G6tgNLis


import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


# 1- download dataset
# 2- create data loader        # class to wrap data to load in batches
# 3- build model
# 4- train
# 5- save trained model


BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.001


class FeedForwardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten() # first layer
        self.dense_layers = nn.Sequential( # Sequential allows to pack multiple layers together
            nn.Linear(28 * 28, 256),   #first argument: number input features, second: number output features
            nn.ReLU(), #activation function
            nn.Linear(256, 10) #output is 10 because 10 classes in MNIST dataset
        )
        self.softmax = nn.Softmax(dim=1) #final layer, will take all vals and transform them so that sum becomes 1 (normalization)

    def forward(self, input_data): # how to manipulate data
        x = self.flatten(input_data)
        logits = self.dense_layers(x) # outputs
        predictions = self.softmax(logits) # passes logits and returns normaized predictions
        return predictions

# just a default dataset as test
def download_mnist_datasets():
    train_data = datasets.MNIST( # MNIST is a dataset class
        root="data", # where to store the downloaded data, foldername
        train=True,
        download=True, # because not downloaded yet
        transform=ToTensor(), #transformation to dataset to tensors
    )
    validation_data = datasets.MNIST(
        root="data",
        train=False, # give the non-train part of dataset
        download=True,
        transform=ToTensor(),
    )
    return train_data, validation_data


def create_data_loader(train_data, batch_size):
    train_dataloader = DataLoader(train_data, batch_size=batch_size)
    return train_dataloader


def train_single_epoch(model, data_loader, loss_fn, optimiser, device):
    for input, target in data_loader:
        input, target = input.to(device), target.to(device)

        # calculate loss
        prediction = model(input)
        loss = loss_fn(prediction, target)

        # backpropagate error and update weights
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    print(f"loss: {loss.item()}")


def train(model, data_loader, loss_fn, optimiser, device, epochs):
    for i in range(epochs):
        print(f"Epoch {i+1}")
        train_single_epoch(model, data_loader, loss_fn, optimiser, device)
        print("---------------------------")
    print("Finished training")


if __name__ == "__main__":

    # download data and create data loader
    train_data, _ = download_mnist_datasets()
    train_dataloader = create_data_loader(train_data, BATCH_SIZE)   #load in batches

    # construct model and assign it to device
    if torch.cuda.is_available():
        device = "cuda" #gpu acceleration available
    else:
        device = "cpu"
    print(f"Using {device}")
    feed_forward_net = FeedForwardNet().to(device)
    print(feed_forward_net)

    # initialise loss funtion + optimiser
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.Adam(feed_forward_net.parameters(),
                                 lr=LEARNING_RATE)

    # train model
    train(feed_forward_net, train_dataloader, loss_fn, optimiser, device, EPOCHS)

    # save model
    torch.save(feed_forward_net.state_dict(), "feedforwardnet.pth")
    print("Trained feed forward net saved at feedforwardnet.pth")