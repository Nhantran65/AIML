#import Libraries
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
digits = train_df.drop(columns=['label']).values
digits = digits / 255.
label = train_df['label'].values

digits.max(), label.max()

# Show 25 digits of data
fig, axis = plt.subplots(5, 4, figsize=(22, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(digits[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Real digit is {}".format(label[i]))
    

# Part 2 Machine learning
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss

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
# Model accuracy for Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
rf_acc = accuracy_score(y_test, y_pred)
cross_entropy_loss = log_loss(y_test, model.predict_proba(X_test))

print("Random Forest Classifier:")
print("Model accuracy is: {0:.3f}%".format(rf_acc * 100))
print("Cross Entropy Loss: {0:.3f}".format(cross_entropy_loss))



# Compare our result
fig, axis = plt.subplots(5, 5, figsize=(18, 20))

for i, ax in enumerate(axis.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='binary')
    ax.set(title = "Predicted digit {0}\nTrue digit {1}".format(y_pred[i], y_test[i]))
    


#Confusion matrix
np.unique(y_test, return_counts=True)

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(12, 8))
sns.heatmap(cm, annot=True, fmt='.0f')
plt.xlabel("Predicted Digits")
plt.ylabel("True Digits")
plt.show()

# Diffirent Algorithm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB

# Model accuracy for Naive Bayes
gnb = MultinomialNB(alpha=1e-3)
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
gnb_acc = accuracy_score(y_test, y_pred)
cross_entropy_loss = log_loss(y_test, gnb.predict_proba(X_test))

print("Naive Bayes:")
print("NB accuracy is: {0:.3f}%".format(gnb_acc * 100))
print("NB Cross Entropy Loss: {0:.3f}".format(cross_entropy_loss))

# Submission and result
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

# Relevant metrics for the cases(s)
import matplotlib.pyplot as plt

# Calculate accuracy and cross-entropy loss for Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=0)
rf_model.fit(X_train, y_train)
rf_y_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_y_pred)
rf_cross_entropy_loss = log_loss(y_test, rf_model.predict_proba(X_test))

# Calculate accuracy and cross-entropy loss for Naive Bayes
gnb = MultinomialNB(alpha=1e-3)
gnb.fit(X_train, y_train)
gnb_y_pred = gnb.predict(X_test)
gnb_acc = accuracy_score(y_test, gnb_y_pred)
gnb_cross_entropy_loss = log_loss(y_test, gnb.predict_proba(X_test))

# Create a bar chart to display the metrics
models = ['Random Forest', 'Naive Bayes']
accuracy_scores = [rf_acc, gnb_acc]
cross_entropy_losses = [rf_cross_entropy_loss, gnb_cross_entropy_loss]

plt.figure(figsize=(10, 6))
plt.bar(models, accuracy_scores, color='blue', alpha=0.7, label='Accuracy')
plt.bar(models, cross_entropy_losses, color='red', alpha=0.7, label='Cross-Entropy Loss')
plt.xlabel('Models')
plt.ylabel('Metrics')
plt.title('Model Metrics for Random Forest and Naive Bayes')
plt.legend()
plt.show()