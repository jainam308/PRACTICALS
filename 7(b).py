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