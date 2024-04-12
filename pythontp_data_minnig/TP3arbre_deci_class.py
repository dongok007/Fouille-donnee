import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

# Load breast cancer data
data = load_breast_cancer()

# Define features
features = ['mean radius', 'mean concave points']
indices = [list(data.feature_names).index(feature) for feature in features]
X = data.data[:, indices]
y = data.target

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic Regression Model
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

# Decision Tree Classifier Model
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred_tree = clf.predict(X_test)

# Visualization for Logistic Regression
plt.figure(figsize=(12, 6))
correct = y_test == y_pred
plt.scatter(X_test[correct, 0], X_test[correct, 1], c='green', marker='o', edgecolor='black', label='Correct (Logistic)')

incorrect = ~correct
plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='red', marker='x', label='Incorrect (Logistic)')

# Visualization for Decision Tree
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Paired)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title('Decision Regions - Decision Tree')
plt.xlabel('Mean Radius')
plt.ylabel('Mean Concave Points')
plt.legend()
plt.show()

# Print additional information (optional)
print("Logistic Regression Accuracy:", log_reg.score(X_test, y_test))
print("Decision Tree Accuracy:", clf.score(X_test, y_test))

# Print limits on x and y axes
print("Limites sur l'axe des x : (", x_min, ",", x_max, ")")
print("Limites sur l'axe des y : (", y_min, ",", y_max, ")")
