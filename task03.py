# Importing the required libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

#please use googlecollab for execution
# Upload the dataset
uploaded = files.upload()

# Assuming the file is uploaded as a CSV
# Load the dataset
df = pd.read_csv(next(iter(uploaded)))

# Display the first few rows of the dataset
print(df.head())

# Scatter plot between SepalLength and SepalWidth
plt.figure(figsize=(8,6))
sns.scatterplot(x='SepalLength', y='SepalWidth', data=df)
plt.title('Relationship between SepalLength and SepalWidth')
plt.show()

# Scatter plot between PetalLength and PetalWidth
plt.figure(figsize=(8,6))
sns.scatterplot(x='PetalLength', y='PetalWidth', data=df)
plt.title('Relationship between PetalLength and PetalWidth')
plt.show()

# Histogram for SepalLength, SepalWidth, PetalLength, and PetalWidth
plt.figure(figsize=(14,10))
for i, column in enumerate(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'], 1):
    plt.subplot(2, 2, i)
    sns.histplot(df[column], kde=True)
    plt.title(f'Histogram of {column}')
plt.tight_layout()
plt.show()

# Violin plots for SepalLength, SepalWidth, PetalLength, and PetalWidth
plt.figure(figsize=(14,10))
for i, column in enumerate(['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth'], 1):
    plt.subplot(2, 2, i)
    sns.violinplot(y=df[column])
    plt.title(f'Violin plot of {column}')
plt.tight_layout()
plt.show()


# K-Nearest Neighbors Classifier

# Assuming the dataset has a target column named 'Species' (or you can adjust based on your dataset)
X = df[['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']]
y = df['Species']  # Replace with the actual target column if different

# Splitting the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Make predictions
y_pred = knn.predict(X_test)

# Evaluate the model
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:\n', classification_report(y_test, y_pred))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
