# Classification-PyTorch
Here we have a single Python file containing two models and two datasets. <br>
Both models are created using PyTorch. For each iteratioin we go through the data we check for the best performing "version" of the model, and we save the state dict for later use.<br>
 <br>
Dataset 1: [Make moons, dataset with two classes.](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_moons.html) <br>
Model 1: Binary classification with two input values (2d cordinates), two hidden layers and one output value.  <br>
 <br>
 <b>Here's what the model 1 outputs. </b><br>
<i>Epoch: 0 | Train Loss: 0.68241, Train accuracy: 50.00% | Test Loss: 0.61754, Test accuracy: 73.50% <br>
Epoch: 125 | Train Loss: 0.05887, Train accuracy: 97.75% | Test Loss: 0.03780, Test accuracy: 99.00% <br>
Epoch: 250 | Train Loss: 0.05674, Train accuracy: 97.75% | Test Loss: 0.04098, Test accuracy: 99.00% <br>
Epoch: 375 | Train Loss: 0.05521, Train accuracy: 97.62% | Test Loss: 0.03997, Test accuracy: 99.00% <br>
Epoch: 500 | Train Loss: 0.05387, Train accuracy: 97.62% | Test Loss: 0.04159, Test accuracy: 99.00% <br>
Epoch: 625 | Train Loss: 0.05226, Train accuracy: 98.00% | Test Loss: 0.04095, Test accuracy: 99.00% <br>
Epoch: 750 | Train Loss: 0.05123, Train accuracy: 98.00% | Test Loss: 0.05400, Test accuracy: 98.00% <br>
Epoch: 875 | Train Loss: 0.04928, Train accuracy: 97.88% | Test Loss: 0.05075, Test accuracy: 98.00% <br>
*************** <br>
Best saved test loss: 0.03568. Found on epoch: 306. <br>
Model state_dict saved to path: V:\binary_classification_state_dict.pth <br> </i>

 This is a plot of the Dataset 1. And right below it, is another plot with the trained model. There are two plots, one for the training data and one for the testing data. <br>
![image](https://github.com/asuzi/Classification-PyTorch/assets/61744031/da798fce-7136-4a0f-b9ed-a2a674b68968)
![image](https://github.com/asuzi/Classification-PyTorch/assets/61744031/bce054f0-2d38-4aba-ad03-a318cce6854c)

Dataset 2: [Spiral dataset, with multiple classes.](https://cs231n.github.io/neural-networks-case-study/) <br>
Model 2: Multiclass classification with two input values (2d cordinates), two hidden layers and three output values. <br>
 <br>
 <b>Here's what the model 2 outputs. </b><br>
<i>Epoch: 0 | Train Loss: 1.11024, Train accuracy: 32.50% | Test Loss: 1.05927, Test accuracy: 55.00%<br>
Epoch: 125 | Train Loss: 0.03669, Train accuracy: 99.17% | Test Loss: 0.01513, Test accuracy: 100.00%<br>
Epoch: 250 | Train Loss: 0.02230, Train accuracy: 99.17% | Test Loss: 0.00461, Test accuracy: 100.00%<br>
Epoch: 375 | Train Loss: 0.01777, Train accuracy: 99.17% | Test Loss: 0.00202, Test accuracy: 100.00%<br>
Epoch: 500 | Train Loss: 0.01591, Train accuracy: 99.17% | Test Loss: 0.00114, Test accuracy: 100.00%<br>
Epoch: 625 | Train Loss: 0.01501, Train accuracy: 99.17% | Test Loss: 0.00062, Test accuracy: 100.00%<br>
Epoch: 750 | Train Loss: 0.01460, Train accuracy: 99.17% | Test Loss: 0.00036, Test accuracy: 100.00%<br>
Epoch: 875 | Train Loss: 0.01433, Train accuracy: 99.17% | Test Loss: 0.00024, Test accuracy: 100.00%<br>
***************<br>
Best saved test loss: 0.00015. Found on epoch: 975.<br>
Model state_dict saved to path: V:\multiclass_classification_state_dict.pth <br> </i>
This is a plot of the Dataset 2. And right below it, is another plot with the trained model. There are two plots, one for the training data and one for the testing data. <br>
![image](https://github.com/asuzi/Classification-PyTorch/assets/61744031/6c44ebd5-c4ad-4088-a514-c89dd561fe30)
![image](https://github.com/asuzi/Classification-PyTorch/assets/61744031/8775368b-9336-4962-a8d7-1c2a5804904c)
