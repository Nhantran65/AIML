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