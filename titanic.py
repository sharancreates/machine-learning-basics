import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

train_data = pd.read_csv('./datasets/titanic/train.csv')

train_data["Age"] = train_data["Age"].fillna(train_data["Age"].median())
train_data["Embarked"] = train_data["Embarked"].fillna(train_data["Embarked"].mode()[0])

train_data.drop("Cabin", axis=1, inplace=True)

sex = pd.get_dummies(train_data["Sex"], drop_first=True)
embarked = pd.get_dummies(train_data["Embarked"], drop_first=True)
pclass = pd.get_dummies(train_data["Pclass"], drop_first=True).add_prefix("Pclass_")
train_data = pd.concat([train_data, sex, embarked, pclass], axis=1)

train_data.drop(["Sex", "Embarked", "Pclass", "Name", "Ticket", "PassengerId"], axis=1, inplace=True )

X = train_data.drop("Survived", axis=1)
y = train_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

model = LogisticRegression(max_iter=500)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, predictions))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
print("\nClassification Report:\n", classification_report(y_test, predictions))