import numpy as np
original_array = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
flattened_array = original_array.flatten()
max_value = np.max(flattened_array)
min_value = np.min(flattened_array)

# Print the results
print("Original 2D Array:")
print(original_array)

print("\nFlattened Array:")
print(flattened_array)

print("\nMaximum Value of Flattened Array:", max_value)
print("Minimum Value of Flattened Array:", min_value)
