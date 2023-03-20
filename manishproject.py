# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:01:33 2023

@author: Manish Shetty
"""
#PROJECT 1


-----------------
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix

# import the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data'
names = ['id', 'clump_thickness', 'uniformity_of_cell_size', 'uniformity_of_cell_shape', 'marginal_adhesion',
         'single_epithelial_cell_size', 'bare_nuclei', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']
data = pd.read_csv(url, names=names)
data = data.drop('id', axis=1)  



data = data.replace({'?':np.nan})
data = data.dropna() 
data['class'] = data['class'].replace(2, 0) 
data['class'] = data['class'].replace(4, 1)  

# Split the data
X = data.iloc[:, :-1].values 
y = data.iloc[:, -1].values  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Implement logistic regression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
acc_lr = accuracy_score(y_test, y_pred_lr)
cm_lr = confusion_matrix(y_test, y_pred_lr)

# Implement KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn = accuracy_score(y_test, y_pred_knn)
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Implement Naive Bayes 
nb = GaussianNB()
nb.fit(X_train, y_train)
y_pred_nb = nb.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb)
cm_nb = confusion_matrix(y_test, y_pred_nb)

results = pd.DataFrame({'Method': ['Logistic Regression', 'KNN', 'Naive Bayes'],
                        'Accuracy': [acc_lr, acc_knn, acc_nb],
                        'Confusion Matrix': [cm_lr, cm_knn, cm_nb]})
print(results)



#PROJECT 02
import requests
from bs4 import BeautifulSoup
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Scrape news article
url = 'https://monkeylearn.com/sentiment-analysis/'
response = requests.get(url)
soup = BeautifulSoup(response.content, 'html.parser')
text = soup.find_all('p')

# Extract text content
content = ''
for p in text:
    content += p.get_text()

# Perform sentiment analysis
sia = SentimentIntensityAnalyzer()
sentiment = sia.polarity_scores(content)

# Print sentiment results
if sentiment['compound'] > 0:
    print('Positive')
elif sentiment['compound'] < 0:
    print('Negative')
else:
    print('Neutral')





#Project 03
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load the dataset
df = pd.read_csv(r'C:\Users\Manish Shetty\Downloads\CC GENERAL.csv')

# Drop  columns
df.drop(['CUST_ID', 'TENURE'], axis=1, inplace=True)

# Handle missing values
df.fillna(method='ffill', inplace=True)
df.isna().sum()

# Standardize the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Determine the optimal number of clusters
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Perform clustering with KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(scaled_data)

# Add cluster labels to the dataframe
df['Cluster'] = kmeans.labels_

# Print the size of each cluster
print(df['Cluster'].value_counts())

# Plot the clusters
plt.scatter(df['PURCHASES'], df['PAYMENTS'], c=df['Cluster'])
plt.xlabel('Purchases')
plt.ylabel('Payments')
plt.show()
