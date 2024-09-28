# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from google.colab import files
from scipy import stats

#use the Google Collab for execution 
#Take the Titanic data set from kaggale and run the code 
# Upload the file using Colab's file uploader
uploaded = files.upload()

# Read the uploaded CSV file into a pandas DataFrame
# Assuming the file name is the first key of the uploaded dict
file_name = list(uploaded.keys())[0]
df = pd.read_csv(file_name)

# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(df.head())

# Filter only the columns of interest (Age, Fare, Pclass, Ticket, Parch)
columns_of_interest = ['Age', 'Fare', 'Pclass', 'Ticket', 'Parch']
df_filtered = df[columns_of_interest]

# 1. Frequency distribution for discrete variables like Pclass, Parch
print("\nFrequency Distribution for Pclass:")
print(df_filtered['Pclass'].value_counts())

print("\nFrequency Distribution for Parch:")
print(df_filtered['Parch'].value_counts())

# 2. Summary Statistics (Mean, Median, Mode) for numerical columns Age, Fare
print("\nSummary Statistics for Age and Fare:")
summary_statistics = df_filtered[['Age', 'Fare']].describe()
print(summary_statistics)

# Calculate mode for Age and Fare
mode_age = df_filtered['Age'].mode()[0]
mode_fare = df_filtered['Fare'].mode()[0]
print(f"\nMode of Age: {mode_age}")
print(f"Mode of Fare: {mode_fare}")

# Calculate median for Age and Fare
median_age = df_filtered['Age'].median()
median_fare = df_filtered['Fare'].median()
print(f"\nMedian of Age: {median_age}")
print(f"Median of Fare: {median_fare}")

# 3. Measures of Dispersion (Variance, Standard Deviation)
variance_age = df_filtered['Age'].var()
std_dev_age = df_filtered['Age'].std()
variance_fare = df_filtered['Fare'].var()
std_dev_fare = df_filtered['Fare'].std()

print(f"\nVariance of Age: {variance_age}")
print(f"Standard Deviation of Age: {std_dev_age}")
print(f"Variance of Fare: {variance_fare}")
print(f"Standard Deviation of Fare: {std_dev_fare}")

# 4. Interquartile Range (IQR) for Age and Fare
IQR_age = stats.iqr(df_filtered['Age'].dropna())
IQR_fare = stats.iqr(df_filtered['Fare'].dropna())

print(f"\nInterquartile Range of Age: {IQR_age}")
print(f"Interquartile Range of Fare: {IQR_fare}")

# Optional: Plot histogram for visualizing distributions
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
sns.histplot(df_filtered['Age'].dropna(), kde=True)
plt.title('Age Distribution')

plt.subplot(1, 2, 2)
sns.histplot(df_filtered['Fare'], kde=True)
plt.title('Fare Distribution')

plt.tight_layout()
plt.show()
