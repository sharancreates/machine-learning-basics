from sklearn import datasets
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB

dataset = datasets.load_iris ()
model = GaussianNB ()

model.fit (dataset.data, dataset.target)
expected = dataset.target

predicted = model.predict (dataset.data)
print (metrics.classification_report (expected, predicted) )
print (metrics.confusion_matrix (expected, predicted) )

#out of  50 : no wrong classification
#out of 50: 47 right
#out of 50: 47 right