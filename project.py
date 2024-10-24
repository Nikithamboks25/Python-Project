# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('student_sleep_patterns.csv') # Load dataset 

print("Dataset Information:")
print(df.info())  # Displays number of rows, columns, and data types
print("\nDataset Summary Statistics:")
print(df.describe())  # Displays summary statistics

# Data Cleaning:

# 1. Handle missing data
print("\nChecking for Missing Values:")
print(df.isnull().sum())  # Check missing data in each column

# Fill missing values for numeric columns with column mean
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 2. Remove duplicates
df.drop_duplicates(inplace=True)

# 3. Ensure data types are appropriate
print("\nData Types after Cleaning:")
print(df.dtypes)

# Data Exploration:
print("\nDescriptive Statistics of the DataFrame:")
print(df.describe())

# Visualizing distributions of key numeric variables
plt.figure(figsize=(10, 6))
sns.histplot(df['Age'], bins=10, kde=True)
plt.title('Age Distribution')
plt.show()

plt.figure(figsize=(10, 6))
sns.histplot(df['Sleep_Duration'], bins=10, kde=True)
plt.title('Sleep Duration Distribution')
plt.show()

# Exploring relationships using scatter plots
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Study_Hours', y='Sleep_Duration', data=df)
plt.title('Relationship Between Study Hours and Sleep Duration')
plt.show()

# Correlation matrix to examine relationships between numeric variables
plt.figure(figsize=(12, 8))
corr = df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Data Manipulation:
low_sleep_students = df[df['Sleep_Duration'] < 6]
print("\nStudents with Sleep Duration less than 6 hours:")
print(low_sleep_students.head())

# Grouping: Average Sleep Quality by Gender
avg_sleep_quality_by_gender = df.groupby('Gender')['Sleep_Quality'].mean()
print("\nAverage Sleep Quality by Gender:")
print(avg_sleep_quality_by_gender)

# Creating new columns:
df['Sleep_Efficiency'] = df['Sleep_Duration'] / ((df['Weekday_Sleep_End'] - df['Weekday_Sleep_Start']) + (df['Weekend_Sleep_End'] - df['Weekend_Sleep_Start'])) * 2

print("\nSample of DataFrame with New Sleep Efficiency Column:")
print(df.head())

# Data Analysis:
from scipy.stats import ttest_ind

# Split data into two groups: high caffeine intake vs low/no caffeine intake
high_caffeine = df[df['Caffeine_Intake'] > df['Caffeine_Intake'].mean()]
low_caffeine = df[df['Caffeine_Intake'] <= df['Caffeine_Intake'].mean()]

# Perform a t-test to compare the sleep quality of the two groups
t_stat, p_val = ttest_ind(high_caffeine['Sleep_Quality'], low_caffeine['Sleep_Quality'])
print("\nT-Test Results for Caffeine Intake and Sleep Quality:")
print(f"T-statistic: {t_stat}, P-value: {p_val}")

# Pivot table to compare average study hours across university years
pivot_table_study = df.pivot_table(values='Study_Hours', index='University_Year', aggfunc='mean')
print("\nPivot Table - Average Study Hours by University Year:")
print(pivot_table_study)

# Data Visualization:
plt.figure(figsize=(10, 6))
sns.boxplot(x='University_Year', y='Sleep_Duration', hue='Gender', data=df)
plt.title('Sleep Duration by University Year and Gender')
plt.show()

# Bar plot of physical activity by university year
plt.figure(figsize=(10, 6))
sns.barplot(x='University_Year', y='Physical_Activity', data=df)
plt.title('Physical Activity by University Year')
plt.show()

# Save the manipulated DataFrame to a new CSV file
df.to_csv('cleaned_student_data.csv', index=False)

print("\nAnalysis Completed. The cleaned dataset has been saved as 'cleaned_student_data.csv'.")
