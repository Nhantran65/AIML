# imports
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import matplotlib.pyplot as plt
import seaborn as sns

import os
print(os.listdir("./Data"))

# Set our train and test date
train_df = pd.read_csv('./Data/train.csv')
test_df = pd.read_csv('./Data/test.csv')

train_df.head()

# data size
train_df.shape

# show digits distribution
train_df.label.value_counts()

# Set features and label for showing
digits = train_df.drop('label', axis=1).values

digits = digits / 255.
label = train_df['label'].values

digits.max(), label.max()


# Show 25 digits of data
fig, axis = plt.subplots(5, 4, figsize=(22, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(digits[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Real digit is {}".format(label[i]))
    
# Machine Learning
from sklearn.model_selection import train_test_split

# models
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

from sklearn.metrics import confusion_matrix

# Set X, y for fiting
X = digits
y = label

# split data into 90% training and 10% for test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Seting our model
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)

y_pred = model.predict(X_test) # predict our file test data
rf_acc = accuracy_score(y_test, y_pred)

from Intro import train_df, digits, label, plt, RandomForestClassifier, accuracy_score, train_test_split
print("Model accuracy is: {0:.3f}%".format(rf_acc * 100))

# Compare our result
fig, axis = plt.subplots(5, 4, figsize=(18, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Predicted digit {0}\nTrue digit {1}".format(y_pred[i], y_test[i]))
    

np.unique(y_test, return_counts=True)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='.0f')
plt.xlabel("Predicted Digits")
plt.ylabel("True Digits")
plt.show()

test_X = test_df.values / 255.
rfc_pred = model.predict(test_X)
gnb_pred = gnb.predict(test_X)


sub = pd.read_csv('./Data/sample_submission.csv')
sub.head()

# Make submission file
sub['Label'] = rfc_pred
sub.to_csv('submission.csv', index=False)

# Make NB submission file
sub['Label'] = gnb_pred
sub.to_csv('GNB_submission.csv', index=False)

# Show our submission file
sub.head(10)

