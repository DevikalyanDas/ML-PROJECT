import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import preprocessing

#Read the data
df=pd.read_csv('news.csv')
#Get shape and head
df.shape
df.head()

#Get the labels
labels=df.label
labels.head()
#print(df['text'])

#classes = df['label'].unique()

#Split the dataset
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)

#Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)

#Initialize a PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
y_pred=pac.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

#Build confusion matrix
cm=confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])

#plot_confusion_matrix(y_test,y_pred,labels,normalize=False,title=None,cmap=plt.cm.Blues)

# Plot non-normalized confusion matrix

plot_confusion_matrix(y_test, y_pred, classes=labels,title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y_test, y_pred, classes=labels, normalize=True,title='Normalized confusion matrix')

plt.show()
