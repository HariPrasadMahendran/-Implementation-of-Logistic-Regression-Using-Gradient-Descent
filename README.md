# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. 
2. 
3. 
4. 

## Program:
```
M.Hari Prasad(25013933)
/*
import pandas as pd
import numpy as np


data = pd.read_csv(r"C:\Users\acer\Downloads\Placement_Data.csv")
data1 = data.copy()
data1 = data1.drop(['sl_no', 'salary'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])


X = data1.iloc[:, :-1].values
y = data1["status"].values

# -----------------------------
# Add Bias Term
# -----------------------------
X = np.c_[np.ones(X.shape[0]), X]

# -----------------------------
# Initialize theta
# -----------------------------
theta = np.random.randn(X.shape[1])

# -----------------------------
# Sigmoid Function
# -----------------------------
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# -----------------------------
# Loss Function
# -----------------------------
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.mean(y * np.log(h + 1e-9) + (1 - y) * np.log(1 - h + 1e-9))

# -----------------------------
# Gradient Descent
# -----------------------------
def gradient_descent(theta, X, y, alpha, iterations):
    m = len(y)
    for _ in range(iterations):
        h = sigmoid(X.dot(theta))
        gradient = (X.T.dot(h - y)) / m
        theta = theta - alpha * gradient
    return theta

# -----------------------------
# Train Model
# -----------------------------
theta = gradient_descent(theta, X, y, alpha=0.01, iterations=1000)

# -----------------------------
# Prediction Function
# -----------------------------
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    return np.where(h >= 0.5, 1, 0)

# -----------------------------
# Predictions on Training Data
# -----------------------------
y_pred = predict(theta, X)

# -----------------------------
# Accuracy
# -----------------------------
accuracy = np.mean(y_pred == y)
print("Accuracy:", accuracy)

print("Predicted:\n", y_pred)
print("Actual:\n", y)

# -----------------------------
# New Data Prediction
# -----------------------------
# NOTE: same order as training features (WITHOUT bias)
xnew_features = np.array([[0, 87, 0, 95, 0, 2, 78, 2, 0, 0, 1, 0]])

# Add bias
xnew = np.c_[np.ones(xnew_features.shape[0]), xnew_features]

y_prednew = predict(theta, xnew)
print("Predicted Result:", y_prednew)
  
*/
```

## Output:
<img width="892" height="388" alt="Screenshot 2026-02-13 152627" src="https://github.com/user-attachments/assets/233b546f-194d-432a-aee0-da02e0cfed3b" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

