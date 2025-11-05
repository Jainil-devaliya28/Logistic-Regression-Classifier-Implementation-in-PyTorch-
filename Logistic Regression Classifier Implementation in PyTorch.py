import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class LogisticTorch:
    
    def __init__(self, lr=0.01, epochs=1000, device=None, random_state=42):
        
        self.lr = lr
        self.epochs = epochs
        self.model = None
        self.loss_history = []
        self.X_mean = None
        self.X_std = None
        self.random_state = random_state
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        torch.manual_seed(self.random_state)

    def _initialize_model(self, n_features):
        self.model = nn.Linear(n_features, 1).to(self.device)
        self.sigmoid = nn.Sigmoid()
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr)

    def _preprocess_data(self, X, is_fit=False):
        X_tensor = torch.from_numpy(X).float()
        if is_fit:
            self.X_mean = X_tensor.mean(dim=0, keepdim=True)
            self.X_std = X_tensor.std(dim=0, keepdim=True)
            self.X_std[self.X_std < 1e-6] = 1.0
        X_tensor = (X_tensor - self.X_mean) / self.X_std
        return X_tensor.to(self.device)

    def fit(self, X, y):
        X_tensor = self._preprocess_data(X, is_fit=True)
        y_tensor = torch.from_numpy(y).float().view(-1, 1).to(self.device)
        n_samples, n_features = X.shape
        self._initialize_model(n_features)
        self.loss_history = []
        for epoch in range(self.epochs):
            logits = self.model(X_tensor)
            loss = self.criterion(logits, y_tensor)
            self.loss_history.append(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Model is not trained yet. Call fit() first.")
        X_tensor = self._preprocess_data(X)
        with torch.no_grad():
            logits = self.model(X_tensor)
            probabilities = self.sigmoid(logits)
        proba_class1 = probabilities.cpu().numpy().flatten()
        proba_class0 = 1 - proba_class1
        return np.column_stack((proba_class0, proba_class1))

    def predict(self, X):
        probabilities = self.predict_proba(X)[:, 1]
        return (probabilities >= 0.5).astype(int)

## Data Preparation and Model Training

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Train LogisticTorch
lr_torch = LogisticTorch(lr=0.05, epochs=1000) 
lr_torch.fit(X, y)

# Train scikit-learn Logistic Regression
lr_sklearn = LogisticRegression(solver='lbfgs', C=1e5, random_state=42)
lr_sklearn.fit(X, y)

## Performance Comparison and Metrics

# Predict and calculate accuracy for LogisticTorch
y_pred_torch = lr_torch.predict(X)
accuracy_torch = accuracy_score(y, y_pred_torch)

# Predict and calculate accuracy for scikit-learn
y_pred_sklearn = lr_sklearn.predict(X)
accuracy_sklearn = accuracy_score(y, y_pred_sklearn)

print("## Model Performance Metrics")
print(f"LogisticTorch Accuracy:   {accuracy_torch:.4f}")
print(f"Scikit-learn Accuracy:    {accuracy_sklearn:.4f}")

## Plotting

def plot_decision_boundary(model, X, y, title):
    h = .02  
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00'])

    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

  
    plt.figure(figsize=(6, 5))
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(title)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.show()

## Plot 1: Decision Boundary
print("\n## Decision Boundary Plots")
plot_decision_boundary(lr_torch, X, y, "LogisticTorch Decision Boundary (PyTorch)")
plot_decision_boundary(lr_sklearn, X, y, "Scikit-learn Logistic Regression Decision Boundary")

## Plot 2: Loss Curve
print("\n## Loss Curve Plot")
plt.figure(figsize=(8, 5))
plt.plot(range(len(lr_torch.loss_history)), lr_torch.loss_history, label='Training Loss')
plt.title('Loss Curve for LogisticTorch Training')
plt.xlabel('Epoch')
plt.ylabel('Binary Cross-Entropy Loss')
plt.grid(True)
plt.legend()
plt.show()