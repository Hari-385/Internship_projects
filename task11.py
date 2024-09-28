#task-1
# Step 1: Upload the CSV file in Google Colab
from google.colab import files
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Upload the CSV file
uploaded = files.upload()

# Step 2: Load the CSV file into a pandas DataFrame
filename = list(uploaded.keys())[0]  # Get the name of the uploaded file
df = pd.read_csv(filename)

# Step 3: Display the first few rows of the data
print("Initial Data:")
print(df.head())

# Step 4: Handling missing values
# Filling missing numeric values with the median
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)

# If Ticket is categorical, fill with the most frequent value
df['Ticket'].fillna(df['Ticket'].mode()[0], inplace=True)

# Step 5: Handling outliers using IQR (Interquartile Range) method
def remove_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    cleaned_data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return cleaned_data

# Remove outliers from Age and Fare
df_cleaned = remove_outliers_iqr(df, 'Age')
df_cleaned = remove_outliers_iqr(df_cleaned, 'Fare')

print("\nData After Cleaning:")
print(df_cleaned.describe())

# Step 6: Visualize the cleaned data

# Plot distribution of 'Age' and 'Fare' after cleaning
plt.figure(figsize=(14,6))

plt.subplot(1, 2, 1)
sns.histplot(df_cleaned['Age'], bins=20, kde=True, color='blue')
plt.title('Age Distribution After Cleaning')

plt.subplot(1, 2, 2)
sns.histplot(df_cleaned['Fare'], bins=20, kde=True, color='green')
plt.title('Fare Distribution After Cleaning')

plt.show()

# Step 7: Show a few cleaned records
print("\nCleaned Data Sample:")
print(df_cleaned.head())
