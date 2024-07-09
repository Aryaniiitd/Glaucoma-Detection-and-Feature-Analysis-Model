import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns

# Load the CSV file into a Pandas DataFrame with the correct delimiter
file_path = '1.csv'
df = pd.read_csv(file_path, delimiter=';')

# INFO
print("Head of the DataFrame:")
print(df.head())

print(df.shape)
print(df.info())

print("\nSummary Statistics:")
print(df.describe())
print(df.isnull().any())
print(df.isnull().sum())
features = df.iloc[:,:-1]
predictions = df.iloc[:,-1]
print(features.head())
print(predictions.head())
numeric_columns = df.select_dtypes(include=['number']).columns
correlation_matrix = df[numeric_columns].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

X_train,X_test,y_train,y_test = train_test_split(features,predictions,test_size = 0.2,random_state = 42,shuffle = True)
Train_data = pd.concat([X_train,y_train],axis = 'columns',names= ['Final Label','Age','Label G1','Label G2','Label G3','G1 ANRI','G1 NVT'])
print("Train data head\n",Train_data.head())
#Summary Stat
print(Train_data.describe())


# Visualize the distribution of numerical columns using histograms
numeric_columns = df.select_dtypes(include=['number']).columns
df[numeric_columns].hist(figsize=(10, 8))
plt.suptitle("Histograms of Numerical Columns")
plt.show()


def draw_scatter_plot(data, x_col, y_col, title):
  plt.figure(figsize=(8, 6))
  for label in data['Label G1'].unique():
    plt.scatter(data[data['Label G1'] == label][x_col], data[data['Label G1'] == label][y_col], label=label)
  plt.xlabel(x_col)
  plt.ylabel(y_col)
  plt.title(title)
  plt.legend()
  plt.grid(True)
  plt.show()

# Draw scatter plots for each feature vs. Label G1
for col in ['Age', 'G1 ANRI', 'G1 NVT']:
  draw_scatter_plot(Train_data, col, 'Label G1', f"Scatter Plot: {col} vs. Label G1")