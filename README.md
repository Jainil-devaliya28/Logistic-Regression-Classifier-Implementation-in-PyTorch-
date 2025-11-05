# Logistic Regression in PyTorch [Problem Statement]

Implement logistic regression from scratch in PyTorch with an interface similar to scikit-learn’s `LogisticRegression`. Your implementation should support the following:

```python
class LogisticTorch:
    def __init__(self, lr=0.01, epochs=1000):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass

    def predict_proba(self, X):
        pass
```

Use the following dataset:

```python
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
```

* Train your ```LogisticTorch``` classifier on this dataset.
* Compare the performance with ```sklearn.linear_model.LogisticRegression```.
* Plot the decision boundary for both models.
* Plot the loss curve during training.
* Report accuracy on the dataset for both models.

# Model Performance Metrics
LogisticTorch Accuracy:   0.8350

Scikit-learn Accuracy:    0.8350
