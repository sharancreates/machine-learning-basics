import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score,confusion_matrix,classification_report,roc_auc_score,roc_curve)

import matplotlib.pyplot as plt
import seaborn as sns

data = load_breast_cancer()

X = pd.DataFrame(data.data,columns=data.feature_names)

y = pd.Series(data.target,name="target")

print("Dataset Shape:", X.shape)
print("\nTarget Distribution:")
print(y.value_counts())

print("\nClass Names:")
print(data.target_names)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42,)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000,random_state=42)

model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:")
print(round(accuracy, 4))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

roc_auc = roc_auc_score(y_test, y_prob)

print("\nROC-AUC Score:")
print(round(roc_auc, 4))


plt.figure(figsize=(6, 5))

sns.heatmap( cm, annot=True,fmt='d',cmap='Blues',xticklabels=data.target_names,yticklabels=data.target_names)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

fpr, tpr, thresholds = roc_curve(y_test,y_prob)

plt.figure(figsize=(7, 5))

plt.plot(fpr,tpr,label=f"AUC = {roc_auc:.3f}")

plt.plot([0, 1],[0, 1],linestyle="--")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()

plt.show()

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": model.coef_[0]
})

coefficients["Abs_Coefficient"] = np.abs( coefficients["Coefficient"])

coefficients = coefficients.sort_values(by="Abs_Coefficient",ascending=False)

print("\nTop 15 Most Important Features:")
print(coefficients.head(15))

top_features = coefficients.head(15)

plt.figure(figsize=(10, 6))

sns.barplot(data=top_features,x="Coefficient",y="Feature")

plt.title("Top Logistic Regression Coefficients")
plt.tight_layout()
plt.show()

sample = X.iloc[[0]]

sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)[0]

probability = model.predict_proba(sample_scaled)[0]

print("\nPrediction for First Sample:")
print("Class:", data.target_names[prediction])

print(f"Probability Malignant: {probability[0]:.4f}")

print(f"Probability Benign: {probability[1]:.4f}")