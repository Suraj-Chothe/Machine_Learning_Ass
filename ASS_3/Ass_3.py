# Sir code
#import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#reading Dataset
dataset=pd.read_csv("DT-Data.csv")
X=dataset.iloc[:,:-1]
y=dataset.iloc[:,5].values

#Perform Label encoding
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()

X = X.apply(LabelEncoder().fit_transform)
print (X)

from sklearn.tree import DecisionTreeClassifier
regressor=DecisionTreeClassifier()
regressor.fit(X.iloc[:,1:5],y)

#Predict value for the given expression
X_in=np.array([0,1,0,1])

y_pred=regressor.predict([X_in])
print ("Prediction:", y_pred)

from six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
# Create DOT data
dot_data = StringIO()

export_graphviz(regressor, out_file=dot_data, filled=True, rounded=True, special_characters=True)
# Draw graph
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('Decision_Tree.png')
# Show graph
Image(graph.create_png())







##########
# next code
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.preprocessing import LabelEncoder

# Step 1: Load your dataset from the file
df = pd.read_csv('DT-Data.csv')  # Replace with the path to your dataset if needed

# Step 2: Initialize LabelEncoder
le = LabelEncoder()

# Step 3: Apply LabelEncoder to categorical columns
df['age'] = le.fit_transform(df['age'])
df['income'] = le.fit_transform(df['income'])
df['gender'] = le.fit_transform(df['gender'])
df['marital_status'] = le.fit_transform(df['marital_status'])
df['buys'] = le.fit_transform(df['buys'])

# Step 4: Split data into features (X) and target (y)
X = df[['age', 'income', 'gender', 'marital_status']]  # Features
y = df['buys']  # Target variable

# Step 5: Initialize and train the Decision Tree Classifier
tree = DecisionTreeClassifier(criterion='entropy')
tree.fit(X, y)

# Step 6: Display the tree structure
print("Decision Tree Structure:")
print(export_text(tree, feature_names=['age', 'income', 'gender', 'marital_status']))

# Step 7: Handle unseen test data properly using the fitted LabelEncoder
# Test data: [Age < 21, Income = Low, Gender = Female, Marital Status = Married]
test_data = {
    'age': '<21',
    'income': 'Low',
    'gender': 'Female',
    'marital_status': 'Married'
}

# Ensure that the test data is transformed using the encoder fitted on the entire dataset
encoded_test_data = [
    le.transform([test_data['age']])[0],
    le.transform([test_data['income']])[0],
    le.transform([test_data['gender']])[0],
    le.transform([test_data['marital_status']])[0]
]

# Predict for the encoded test data
prediction = tree.predict([encoded_test_data])

# Step 8: Output the prediction
print("\nPrediction for the test data:", le.inverse_transform(prediction))