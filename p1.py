import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'
df = pd.read_csv(url)

# Basic Data Overview
print("Shape of the dataset:", df.shape)
print("Columns and Data Types:\n", df.dtypes)
print("Descriptive Statistics:\n", df.describe(include='all'))
print("Missing Values:\n", df.isnull().sum())

# Univariate Analysis
# Histogram of 'age'
df['age'].hist()
plt.title('Distribution of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Histogram of 'charges'
df['charges'].hist()
plt.title('Distribution of Charges')
plt.xlabel('Charges')
plt.ylabel('Frequency')
plt.show()

# Box Plot for 'charges'
sns.boxplot(x=df['charges'])
plt.title('Box Plot of Charges')
plt.show()

# Bar Chart for 'region'
df['region'].value_counts().plot(kind='bar')
plt.title('Distribution of Region')
plt.xlabel('Region')
plt.ylabel('Count')
plt.show()

# Bivariate Analysis
# Scatter Plot between 'age' and 'charges'
sns.scatterplot(x='age', y='charges', data=df)
plt.title('Age vs Charges')
plt.xlabel('Age')
plt.ylabel('Charges')
plt.show()

# Correlation Matrix and Heatmap
correlation_matrix = df.corr()
print("Correlation Matrix:\n", correlation_matrix)
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Multivariate Analysis
# Facet Grid for 'charges' by 'region'
g = sns.FacetGrid(df, col='region', margin_titles=True)
g.map_dataframe(sns.scatterplot, x='age', y='charges')
g.set_axis_labels('Age', 'Charges')
plt.show()

# Pair Plot for numerical features
sns.pairplot(df[['age', 'bmi', 'charges']])
plt.show()
