# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY
Developing a neural network regression model involves designing a feedforward neural network with fully connected layers to predict continuous numerical values. The model is trained by minimizing a loss function such as Mean Squared Error (MSE), which quantifies the difference between predicted and actual values. Optimization algorithms like RMSprop or Adam are used to adjust the model's weights during training, enabling it to learn patterns in the data and improve its predictions over time.

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: Keerthana S
### Register Number: 212222230066
```python
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        #Include your code here



# Initialize the Model, Loss Function, and Optimizer



def train_model(ai_brain, X_train, y_train, criterion, optimizer, epochs=2000):
    #Include your code here



```
## Dataset Information

![image](https://github.com/user-attachments/assets/b81c2844-f926-4fc1-b38a-5d02233d81d7)


## OUTPUT

### Training Loss Vs Iteration Plot

![Screenshot 2025-02-27 143910](https://github.com/user-attachments/assets/337adeee-d337-4df2-8004-9d964de56664)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/fa26358e-38ed-4322-8a17-b77294314588)


## RESULT

Include your result here
