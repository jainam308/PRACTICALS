


#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Pract-6_1
import numpy as np
import pandas as pd

numpy_array = np.array([1, 2, 3, 4, 5])

series = pd.Series(numpy_array)

print(series)


# In[3]:


# Pract-6_2
import pandas as pd

data = {'A': [1, 2, 3, 4, 5],
        'B': ['apple', 'banana', 'cherry', 'date', 'berry']}
df = pd.DataFrame(data)

first_column_series = df.iloc[:, 0]

print(first_column_series)
# In[4]:


# Pract-6_3
import pandas as pd

data = [10, 20, 30, 40, 50]
series = pd.Series(data)

mean_value = series.mean()
std_deviation = series.std()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)


# In[5]:


# Pract-6_4
import pandas as pd

data = [30, 10, 50, 20, 40]
series = pd.Series(data)

sorted_series = series.sort_values()

print("Sorted Series (Ascending):")
print(sorted_series)

sorted_series_descending = series.sort_values(ascending=False)

print("\nSorted Series (Descending):")
print(sorted_series_descending)


# In[6]:


# Pract-7_1
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

print(df)


# In[7]:


# Pract-7_2
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

df_sorted = df.sort_values(by='Name', ascending=True)

print(df_sorted)


# In[8]:


# Pract-7_3
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

df = df.drop('City', axis=1)

print(df)


# In[13]:


# Pract-7_4
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 40],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

df.to_csv('pract-7_4.csv', sep='\t', index=False)

print("DataFrame written to 'pract-7_4.csv.csv'")


# In[14]:


#Pract-8
import pandas as pd
import matplotlib.pyplot as plt

# Assuming you have a DataFrame named 'stock_data' with columns: Date, Open, Close
# You can load your data from a CSV file or other sources as needed.

# Sample DataFrame
data = {
    'Date': ['2023-09-01', '2023-09-02', '2023-09-03', '2023-09-04', '2023-09-05'],
    'Open': [100, 102, 105, 103, 108],
    'Close': [102, 104, 106, 105, 110]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Convert the 'Date' column to datetime format (if it's not already)
df['Date'] = pd.to_datetime(df['Date'])

# Filter the DataFrame based on specific date range
start_date = '2023-09-01'
end_date = '2023-09-04'
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Create a line plot of opening and closing prices
plt.figure(figsize=(7, 5))
plt.plot(filtered_df['Date'], filtered_df['Open'], marker='o', label='Open Price')
plt.plot(filtered_df['Date'], filtered_df['Close'], marker='o', label='Close Price')

plt.title('Stock Prices Between {} and {}'.format(start_date, end_date))
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()


# In[17]:


# Pract-9
import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame (replace with your actual data)
data = {
    'Date': ['2023-09-01', '2023-09-02', '2023-09-03', '2023-09-04', '2023-09-05'],
    'Open': [100, 102, 105, 103, 108],
    'High': [105, 108, 110, 107, 112],
    'Low': [98, 100, 103, 101, 106],
    'Close': [102, 104, 106, 105, 110],
    'Adjusted Close': [101.5, 103.5, 105.5, 104.5, 109.5],
    'Volume': [10000, 12000, 9500, 11000, 13500]
}

# Create a DataFrame from the sample data
df = pd.DataFrame(data)

# Convert the 'Date' column to datetime format (if it's not already)
df['Date'] = pd.to_datetime(df['Date'])

# Filter the DataFrame based on specific date range
start_date = '2023-09-01'
end_date = '2023-09-04'
filtered_df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]

# Create a plot of Open, High, Low, Close, Adjusted Closing prices, and Volume
plt.figure(figsize=(7, 5))
plt.plot(filtered_df['Date'], filtered_df['Open'], label='Open', marker='o')
plt.plot(filtered_df['Date'], filtered_df['High'], label='High', marker='o')
plt.plot(filtered_df['Date'], filtered_df['Low'], label='Low', marker='o')
plt.plot(filtered_df['Date'], filtered_df['Close'], label='Close', marker='o')
plt.plot(filtered_df['Date'], filtered_df['Adjusted Close'], label='Adjusted Close', marker='o')
plt.bar(filtered_df['Date'], filtered_df['Volume'], label='Volume', alpha=0.5)

plt.title('Stock Prices and Volume Between {} and {}'.format(start_date, end_date))
plt.xlabel('Date')
plt.ylabel('Price / Volume')
plt.legend()
plt.grid(True)

# Rotate x-axis labels for better readability
plt.xticks(rotation=45)

# Display the plot
plt.tight_layout()
plt.show()


# In[19]:


# Pract-10_1
import pandas as pd

# Create a sample DataFrame with missing values
data = {
    'A': [1, 2, None, 4, 5],
    'B': [None, 2, 3, None, 5],
    'C': [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

# Find missing values in the DataFrame
missing_values = df.isnull()

# Display the DataFrame with missing value information
print("Original DataFrame:")
print(df)

# Drop rows containing missing values
df_cleaned = df.dropna()

# Display the DataFrame after dropping missing values
print("\nDataFrame after dropping missing values:")
print(df_cleaned)
# In[20]:


# Pract-10_2
import pandas as pd

# Create a sample DataFrame with duplicate rows
data = {
    'A': [1, 2, 2, 3, 4, 4],
    'B': ['apple', 'banana', 'banana', 'cherry', 'date', 'date']
}

df = pd.DataFrame(data)

# Display the original DataFrame with duplicate rows
print("Original DataFrame:")
print(df)

# Remove duplicate rows
df_cleaned = df.drop_duplicates()

# Display the DataFrame after removing duplicates
print("\nDataFrame after removing duplicates:")
print(df_cleaned)


# In[21]:


# Pract-11
import pandas as pd
import numpy as np

# Create a sample DataFrame with missing values
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [None, 2, 3, None, 5],
    'C': [1, 2, 3, 4, 5]
}

df = pd.DataFrame(data)

# Filter columns where all entries are present
df_filtered_columns = df.dropna(axis=1, how='any')

# Display the DataFrame with only columns where all entries are present
print("DataFrame with columns where all entries are present:")
print(df_filtered_columns)

# Check which rows and columns have NaN values
rows_with_nan = df[df.isnull().any(axis=1)]
columns_with_nan = df.columns[df.isnull().any()]

# Display rows and columns with NaN values
print("\nRows with NaN values:")
print(rows_with_nan)
print("\nColumns with NaN values:")
print(columns_with_nan)

# Drop rows containing any NaNs
df_cleaned = df.dropna()

# Display the DataFrame after dropping rows with NaNs
print("\nDataFrame after dropping rows with NaNs:")
print(df_cleaned)


# In[24]:


# Pract-12
import pandas as pd
from sklearn.datasets import load_iris

# Load a sample dataset from Scikit-learn (Iris dataset in this case)
data = load_iris()

# Convert the dataset into a Pandas DataFrame
df = pd.DataFrame(data.data, columns=data.feature_names)

# Print the keys of the dataset
print("Keys:")
print(data.keys())

# Get the number of rows and columns
num_rows, num_columns = df.shape
print("\nNumber of Rows:", num_rows)
print("Number of Columns:", num_columns)

# Print the feature names
print("\nFeature Names:")
print(data.feature_names)

# Print a description of the dataset
print("\nDataset Description:")
print(data.DESCR)


# In[26]:


# Pract-13

# In[ ]:





# In[ ]:




