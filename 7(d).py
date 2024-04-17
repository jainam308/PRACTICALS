import pandas as pd

# Create a DataFrame
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 22, 28],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston']
}

df = pd.DataFrame(data)

# Write the DataFrame to a CSV file with tab separator
df.to_csv('output_file.csv', sep='\t', index=False)
print("Data written in output_file.csv")