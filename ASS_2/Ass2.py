# Importing libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load the dataset
df = pd.read_csv('iris.csv')

# Step 1: Basic Data Insights
print("First 5 rows of the dataset:\n", df.head())
print("\nSummary Statistics:\n", df.describe())
print("\nCheck for missing values:\n", df.isnull().sum())

# Check for class distribution
print("\nClass Distribution:\n", df['variety'].value_counts())

# Step 2: Univariate Analysis
plt.figure(figsize=(10, 6))
df.drop('variety', axis=1).hist(bins=15, figsize=(10, 6), layout=(2, 2), color='skyblue', edgecolor='black')
plt.suptitle("Univariate Analysis: Histograms of Features", fontsize=16)
plt.show()

# Boxplots for feature distributions by species
plt.figure(figsize=(14, 10))
for i, column in enumerate(df.columns[:-1], 1):
    plt.subplot(2, 2, i)
    sns.boxplot(x='variety', y=column, data=df)
    plt.title(f"Boxplot of {column} by Species")
plt.tight_layout()
plt.show()



# Step 3: Multivariate Analysis
# Pairplot
sns.pairplot(df, hue='variety', diag_kind='kde', palette='Set2')
plt.suptitle("Pairwise Scatter Plots", y=1.02, fontsize=16)
plt.show()


# Heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Heatmap for correlation
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Standardizing the features
scaler = StandardScaler()
X_standardized = scaler.fit_transform(X)


# Applying PCA
pca = PCA(n_components=2)  # Reducing to 2 components for visualization
X_pca = pca.fit_transform(X_standardized)
# Explained variance
explained_variance = pca.explained_variance_ratio_
print("\nExplained variance ratio:", explained_variance)
print("Cumulative explained variance:", explained_variance.cumsum())


# Visualizing PCA results
pca_df = pd.DataFrame(X_pca, columns=['Principal Component 1', 'Principal Component 2'])
pca_df['variety'] = y

plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='Principal Component 1', y='Principal Component 2', hue='variety', palette='Set2')
plt.title("PCA: Scatterplot of Principal Components")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()


'''Together, the first two principal components contain 95.80% of the information.
The first principal component contains 72.77% of the variance and the second principal
component contains 23.03% of the variance. The third and fourth principal component 
contained the rest of the variance of the dataset.
'''




##########################################
# sir cha code

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
# %matplotlib inline

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url
                 , names=['sepal length','sepal width','petal length','petal width','target'])
df.head()
#downloaded = drive.CreateFile({'id':'12BY34aCbYLoLjy3gDUMrZEBUf7l5FZsd'}) # replace the id with id of file you want to access
#downloaded.GetContentFile('iris.csv')
#dataset=pd.read_csv("iris.csv")
#dataset
#https://drive.google.com/file/d/12BY34aCbYLoLjy3gDUMrZEBUf7l5FZsd/view?usp=share_link



features = ['sepal length', 'sepal width', 'petal length', 'petal width']
x = df.loc[:, features].values


y = df.loc[:,['target']].values


x = StandardScaler().fit_transform(x)


pd.DataFrame(data = x, columns = features).head()


#PCA Projection to 2D
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
principalDf.head(5)

df[['target']].head()

finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
finalDf.head(5)


fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)


targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()


#Explained Variance:The explained variance tells us how much information (variance)
#can be attributed to each of the principal components.
pca.explained_variance_ratio_