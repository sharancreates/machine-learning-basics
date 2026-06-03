import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

iris = load_iris()

X = iris.data
Y = iris.target

dataset = pd.DataFrame(X, columns=iris.feature_names)
dataset["class"] = Y

print("Dataset shape:", dataset.shape)
print(dataset.head())

print("\nSummary statistics:")
print(dataset.describe())

print("\nClass distribution:")
print(dataset.groupby("class").size())

#visualisation
features = dataset.iloc[:, 0:4]

features.plot(kind="box", subplots=True, layout=(2, 2), sharex=False, sharey=False)
plt.show()

features.hist()
plt.show()

scatter_matrix(features)
plt.show()

#train-test split
validation_size = 0.20
seed = 6

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=validation_size, random_state=seed, shuffle=True)

#cross validation and model comparison
models = [
    ("LR", LogisticRegression(max_iter=200)),
    ("LDA", LinearDiscriminantAnalysis()),
    ("KNN", KNeighborsClassifier()),
    ("CART", DecisionTreeClassifier()),
    ("NB", GaussianNB()),
    ("SVM", SVC())
]

results = []
names = []

print("\nModel Comparison (Cross-Validation):")

for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")

    results.append(cv_results)
    names.append(name)

    print(f"{name}: {cv_results.mean():.4f} ({cv_results.std():.4f})")


#best model LDA: training on the entire training set
best_model = LinearDiscriminantAnalysis()
best_model.fit(X_train, Y_train)

#evaluation on the test set
predictions = best_model.predict(X_test)

print("\nTest Set Evaluation:")
print("Accuracy:", accuracy_score(Y_test, predictions))

print("\nConfusion Matrix:")
print(confusion_matrix(Y_test, predictions))

print("\nClassification Report:")
print(classification_report(Y_test, predictions, target_names=iris.target_names))

#sample prediction
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = best_model.predict(sample)

print("\nPredicted class:", iris.target_names[prediction[0]])