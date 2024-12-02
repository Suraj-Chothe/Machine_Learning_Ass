# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load your dataset
df = pd.read_csv('iris.csv')  # Replace 'your_dataset.csv' with the path to your dataset

# Preview the dataset
print("Dataset Head:\n", df.head())

# EDA: Check for missing values
print("\nDataset Info:")
print(df.info())
print("\nMissing Values:\n", df.isnull().sum())

# EDA: Summary statistics
print("\nSummary Statistics:\n", df.describe())

# EDA: Check class distribution
target_column = 'variety'  # Replace 'target' with the actual name of your target column
print("\nClass Distribution:\n", df[target_column].value_counts())

# EDA: Pairplot
sns.pairplot(df, hue=target_column, diag_kind='hist')
plt.show()

# EDA: Correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Preprocessing: Split the dataset into features and target
X = df.drop(columns=['variety'])  # Replace 'target_column' with your actual target column
y = df['variety']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model: Gaussian Naive Bayes
model = GaussianNB()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluation: Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Evaluation: Classification report and accuracy
print("\nClassification Report:\n", classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
