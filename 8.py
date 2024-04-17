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
plt.plot(filtered_df['Date'], filtered_df['Open'], label='Open Price')
plt.plot(filtered_df['Date'], filtered_df['Close'], label='Close Price')

plt.title('Stock Prices Between {} and {}'.format(start_date, end_date))
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)

# Display the plot
plt.tight_layout()
plt.show()

