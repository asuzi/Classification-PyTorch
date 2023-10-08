import torch
from torch import nn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
import os

if torch.cuda.is_available():
    device = "cuda"                             # Set cuda as DEVICE
#    print('CUDA is available for this pc.')
else:
    device = "cpu"                              # Set cpu as DEVICE
#    print('CDA is NOT available for this pc.')

BINARY_MODEL_PATH = f"{os.getcwd()}\\binary_classification_state_dict.pth"
MULTICLASS_MODEL_PATH = f"{os.getcwd()}\\multiclass_classification_state_dict.pth"
EPOCH = 1000
LR = 0.01
BINARY_CLASSIFICATION = True
MULTICLASS_CLASSIFICATION = True

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    if len(torch.unique(y)) > 2:
        y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    else:
        y_pred = torch.round(torch.sigmoid(y_logits))  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct/len(y_pred)) * 100
    return acc

def createData_binary_classification():

    X, y = make_moons(1000,
                      random_state=42,
                      noise=0.2)
    
    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.float32)

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)    
    
    moons = pd.DataFrame({"X1": X[:, 0],
                        "X2": X[:, 1],
                        "label": y})
    
    plt.scatter(x=moons["X1"],
            y=moons["X2"],
            c=moons["label"],
            cmap=plt.cm.viridis)
    plt.show()
    return X_train, X_test, y_train, y_test

def createData_multi_classification():
    # https://cs231n.github.io/neural-networks-case-study/
    N = 100 # number of points per class
    D = 2 # dimensionality
    K = 3 # number of classes
    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype='uint8') # class labels
    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j

    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.viridis)
    plt.show()

    X = torch.from_numpy(X).type(torch.float32)
    y = torch.from_numpy(y).type(torch.long)

    x_train, x_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        shuffle=True)
    
    return x_train, x_test, y_train, y_test

class binary_classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 1)
        self.activation = nn.ReLU()

    def forward(self, x):
        logits = self.linear1(x)
        logits = self.activation(logits)
        logits = self.linear2(logits)
        logits = self.activation(logits)
        logits = self.linear3(logits)
        return logits


class multiclass_classification(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(2, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 3)
        self.activation = nn.ReLU()

    def forward(self, x):
        logits = self.linear1(x)
        logits = self.activation(logits)
        logits = self.linear2(logits)
        logits = self.activation(logits)
        logits = self.linear3(logits)
        return logits


def train_test_binary_classification(x_train, x_test, y_train, y_test):
    best = 99999999

    x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)
    model = binary_classification().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCH):
        model.train()

        logits = model(x_train).squeeze()
        pred_probability = torch.sigmoid(logits)
        pred_label = torch.round(pred_probability)

        loss = loss_fn(logits, y_train)

        optim.zero_grad()
        loss.backward()
        optim.step()

        ### 

        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test).squeeze()
            test_pred_probability = torch.sigmoid(test_logits)
            test_pred_label = torch.round(test_pred_probability)

            test_loss = loss_fn(test_logits, y_test)

        if epoch % int((EPOCH/8)) == 0:
            print(f'Epoch: {epoch} | Train Loss: {loss:.5f}, Train accuracy: {accuracy_fn(y_train, pred_label):.2f}% | Test Loss: {test_loss:.5f}, Test accuracy: {accuracy_fn(y_test, test_pred_label):.2f}%')

        if test_loss < best:
            best = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), BINARY_MODEL_PATH)

    print(f'***************\nBest saved test loss: {best:.5f}. Found on epoch: {best_epoch}. \nModel state_dict saved to path: {BINARY_MODEL_PATH}')

def test_train_multiclass_classification(x_train, x_test, y_train, y_test):
    best = 99999999

    x_train, x_test, y_train, y_test = x_train.to(device), x_test.to(device), y_train.to(device), y_test.to(device)
    model = multiclass_classification().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCH):
        model.train()

        logits = model(x_train)
        pred_probability = torch.softmax(logits, dim=1)
        pred_label = torch.argmax(pred_probability, dim=1)

        loss = loss_fn(logits, y_train)

        optim.zero_grad()
        loss.backward()
        optim.step()

        ###

        model.eval()
        with torch.inference_mode():
            test_logits = model(x_test)
            test_pred_probability = torch.softmax(test_logits, dim=1)
            test_pred_label = torch.argmax(test_pred_probability, dim=1)
        
        test_loss = loss_fn(test_logits, y_test)

        if epoch % int((EPOCH/8)) == 0:
            print(f'Epoch: {epoch} | Train Loss: {loss:.5f}, Train accuracy: {accuracy_fn(y_train, pred_label):.2f}% | Test Loss: {test_loss:.5f}, Test accuracy: {accuracy_fn(y_test, test_pred_label):.2f}%')

        if test_loss < best:
            best = test_loss
            best_epoch = epoch
            torch.save(model.state_dict(), MULTICLASS_MODEL_PATH)

    print(f'***************\nBest saved test loss: {best:.5f}. Found on epoch: {best_epoch}. \nModel state_dict saved to path: {MULTICLASS_MODEL_PATH}')

def binary_load_and_plot_best_model():
    model = binary_classification()
    model.load_state_dict(torch.load(BINARY_MODEL_PATH))
    model.eval()

    # Plot decision boundaries for training and test sets
    # https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, x_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, x_test, y_test)
    plt.show()

def binary_load_and_plot_best_model():
    model = binary_classification()
    model.load_state_dict(torch.load(BINARY_MODEL_PATH))
    model.eval()

    # Plot decision boundaries for training and test sets
    # https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, x_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, x_test, y_test)
    plt.show()

def multiclass_load_and_plot_best_model():
    model = multiclass_classification()
    model.load_state_dict(torch.load(MULTICLASS_MODEL_PATH))
    model.eval()

    # Plot decision boundaries for training and test sets
    # https://github.com/mrdbourke/pytorch-deep-learning/blob/main/helper_functions.py
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, x_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, x_test, y_test)
    plt.show()



if __name__ == "__main__":

    if BINARY_CLASSIFICATION:
        x_train, x_test, y_train, y_test = createData_binary_classification()
        train_test_binary_classification(x_train, x_test, y_train, y_test)
        binary_load_and_plot_best_model()
        
    if MULTICLASS_CLASSIFICATION:
        x_train, x_test, y_train, y_test = createData_multi_classification()
        test_train_multiclass_classification(x_train, x_test, y_train, y_test)
        multiclass_load_and_plot_best_model()
