import pandas as pd

data = [1, 2, 3, 4, 5]
series = pd.Series(data)
print("Orignal Series:\n", series)
mean_value = series.mean()
std_deviation = series.std()

print("Mean:", mean_value)
print("Standard Deviation:", std_deviation)