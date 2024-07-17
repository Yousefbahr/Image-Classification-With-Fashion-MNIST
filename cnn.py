import random

import matplotlib.pyplot as plt
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

# Get data set, FASHION-MNIST
train_data = datasets.FashionMNIST(
    root = "FASHION-MNIST",
    train=True,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

test_data = datasets.FashionMNIST(
    root = "FASHION-MNIST",
    train=False,
    download=True,
    transform=ToTensor(),
    target_transform=None
)

## split the data into mini-batches using a Dataloader
BATCH_SIZE = 32
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

## a batch's shape - > [32, 1, 28, 28]


def get_accuracy(y_true, y_preds):
    correct = torch.eq(y_true, y_preds,).sum().item()
    return (correct / len(y_preds)) * 100


class CNN(nn.Module):
    def __init__(self,input_shape, hidden_units, output_shape):
        super().__init__()
        # A Tiny VGG Architecture
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(in_channels= input_shape, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels= hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units,
                      kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

        self.classifier_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7 ,
                      out_features=output_shape),
        )

    def forward(self, x):
        return self.classifier_layer(self.conv_block2(self.conv_block1(x)))


cnn = CNN(input_shape=1,
          hidden_units=10,
          output_shape=len(train_data.classes))

cnn_optim = torch.optim.SGD(params=cnn.parameters(), lr=0.01)
loss_fn = nn.CrossEntropyLoss()


# Training
epochs = 3
for epoch in range(epochs):
    print(f"Epoch: {epoch}/{epochs}\n-------")
    train_loss, train_acc = 0, 0

    # X here is one batch consisting of 32 images (BATCH_SIZE)
    # y is labels for the 32 images in the batch
    for batch, (X, y) in enumerate(train_dataloader):
        cnn.train()
        # get predictions
        y_pred = cnn(X)
        # calculate loss per batch
        loss = loss_fn(y_pred, y)
        # add each batch's loss to get train loss per epoch
        train_loss += loss
        train_acc += get_accuracy(y, y_pred.argmax(dim=1))

        cnn_optim.zero_grad()

        loss.backward()
        # update weights for each batch
        cnn_optim.step()

        if batch % 650 == 0:
            print(f"Looked at {batch * BATCH_SIZE / len(train_dataloader.dataset):.2f}% of samples")

    # Divide total train loss by number of batches to get train loss by epoch
    train_loss /= len(train_dataloader)
    train_acc /= len(train_dataloader)
    ## Testing loop
    test_loss, test_acc = 0, 0
    cnn.eval()
    with torch.inference_mode():
        for X_test, y_test in test_dataloader:
            test_pred = cnn(X_test)
            test_loss += loss_fn(test_pred, y_test)
            test_acc += get_accuracy(y_test, test_pred.argmax(dim=1))

        # Calulcate test loss and accuracy per epoch
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    print(f"\nTrain loss: {train_loss:.4f} | Train_acc: {train_acc:.4f} |Test Loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")

# Testing
loss, acc = 0, 0
cnn.eval()
with torch.inference_mode():
    for X, y in test_dataloader:
        preds = cnn(X)
        loss += loss_fn(preds, y)
        acc += get_accuracy(y, preds.argmax(dim=1))

    loss /= len(test_dataloader)
    acc /= len(test_dataloader)

print(f"Loss: {loss.item()}, Accuracy: {acc}% ")


# Get a random sample of 9 and check predictions vs target classifications
class_names = train_data.classes
samples, labels = zip(*random.sample(list(test_data), k=9))

# get probabilities of the top 3 classifications
preds = []
pred_labels = []
for sample in samples:
    sample = torch.unsqueeze(sample, dim=0)
    logits = cnn(sample)
    y_preds = torch.softmax(logits.squeeze(), dim=0).sort(descending=True)
    preds.append(y_preds.values[:3])
    pred_labels.append([class_names[index] for index in y_preds.indices[:3]])

# Plot predictions vs target
plt.figure(figsize=(12, 12), label='Top 3 Predictions vs Target')
nrows = 3
ncols = 3
for i, sample in enumerate(samples):
    ax = plt.subplot(nrows, ncols , i + 1)
    plt.imshow(sample.squeeze(), cmap="gray")
    target_label = class_names[labels[i]]

    y = 33
    for j in range(len(pred_labels[0])):
        text = f"{j + 1}) {pred_labels[i][j]} : {preds[i][j] * 100 :.2f}%"
        # if pred same as target, text with green color
        # else red color
        if pred_labels[i][j] == target_label:
            ax.text(0, y, text, fontsize=10, color='green')

        else:
            ax.text(0, y, text, fontsize=10, color="red")

        y += 5

    plt.title(f"Actual: {target_label}", fontsize=10)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout(pad=5, w_pad=5, h_pad=3)

plt.show()
