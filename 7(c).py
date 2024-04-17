import pandas as pd

# Create a DataFrame from a dictionary
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 22, 28],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)
print("After Droping City Column:")
# Delete the 'City' column from the DataFrame
df = df.drop('City', axis=1)

# Display the DataFrame without the 'City' column
print(df)
