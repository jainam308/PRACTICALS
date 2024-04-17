import pandas as pd
series_data = [5,3,1,9,8,7,6]
series = pd.Series(series_data)
print("Original Series:\n", series)
print("Sorted Series:\n", series.sort_values(ascending=True))