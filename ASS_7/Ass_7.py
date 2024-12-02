# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Step 2: Load your dataset
# Replace 'your_dataset.csv' with your file path
data = pd.read_csv('income_evaluation.csv')

# Step 3: Understand the data
print(data.head())
print(data.info())
print(data.describe())

# Step 4: Preprocess the data
# Handle missing values if any
data = data.dropna()

# Convert categorical variables to numeric (if any)
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Step 5: Define features and target
# Replace 'target_column' with the name of your target column
X = data.drop(columns=[' income'])
y = data[' income']

# Step 6: Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 7: Train the Gradient Boosting Classifier
gbc = GradientBoostingClassifier(random_state=42)
gbc.fit(X_train, y_train)

# Step 8: Make predictions
y_pred = gbc.predict(X_test)

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Optional: Feature Importance
feature_importance = pd.Series(gbc.feature_importances_, index=X.columns).sort_values(ascending=False)
print("Feature Importances:")
print(feature_importance)



##############################################################
#sir cha code

# Import all relevant libraries

from sklearn.ensemble import GradientBoostingClassifier

import numpy as np

import pandas as pd

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score, confusion_matrix

from sklearn import preprocessing

import warnings

warnings.filterwarnings("ignore")




#STEP-1: Import Libraries
# Code to read csv file into colaboratory:
#!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials


#STEP-2: Autheticate E-Mail ID
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)



# Get File from Drive using file-ID
downloaded = drive.CreateFile({'id':'1zI-X3zdiuM9u74zQyKIShvAUtPJQ7jUK'}) # replace the id with id of file you want to access
downloaded.GetContentFile('income_evaluation.csv')
# https://drive.google.com/file/d/1zI-X3zdiuM9u74zQyKIShvAUtPJQ7jUK/view?usp=sharing  (Dataset Downloads Link)



#Now let’s read the dataset and look at the columns to understand the information better.
#https://drive.google.com/file/d/1zI-X3zdiuM9u74zQyKIShvAUtPJQ7jUK/view?usp=sharing
df = pd.read_csv('income_evaluation.csv')
df.head()

df.shape
df.info()
df.isnull().sum()

df.columns
#df.drop(columns=' fnlwgt',inplace=True)
df.columns

X = df.drop(columns=' income')
y = df[' income']

from sklearn.preprocessing import LabelEncoder

def label_encoder(a):
    le = LabelEncoder()
    df[a] = le.fit_transform(df[a])
    
    
label_list = [' workclass', ' education',' marital-status',
       ' occupation', ' relationship', ' race', ' sex',' native-country', ' income']


for i in label_list:
    label_encoder(i)
    
df.head()

from sklearn.model_selection import train_test_split

X = df.drop([' income'],axis=1).values   # independant features
y = df[' income'].values					# dependant variable

# Choose your test size to split between training and testing sets:
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)


print("X_train shape:",X_train.shape)
print("y_test shape:",y_test.shape)
print("X_test shape:",X_test.shape)
print("y_train shape:",y_train.shape)

#Buildimg Gradient Boosting Model
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


gradient_booster = GradientBoostingClassifier(learning_rate=0.1)
accuracies = cross_val_score(gradient_booster, X_train, y_train, cv=5)
gradient_booster.fit(X_train,y_train)

print("Train Score:",np.mean(accuracies))
print("Test Score:",gradient_booster.score(X_test,y_test))


result_dict_train = {}
result_dict_test = {}
result_dict_train["Gradient-Boost Default Train Score"] = np.mean(accuracies)
result_dict_test["Gradient-Boost Default Test Score"] = gradient_booster.score(X_test,y_test)


grid = {
    'learning_rate':[0.01,0.05,0.1],
    'n_estimators':np.arange(100,500,100),
}

gb = GradientBoostingClassifier()
gb_cv = GridSearchCV(gb, grid, cv = 4)
gb_cv.fit(X_train,y_train)
print("Best Parameters:",gb_cv.best_params_)
print("Train Score:",gb_cv.best_score_)
print("Test Score:",gb_cv.score(X_test,y_test))

result_dict_train
result_dict_test