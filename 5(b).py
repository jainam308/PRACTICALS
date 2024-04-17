import numpy as np

original_array = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]])
mean_values = np.mean(original_array, axis=1)

std_deviation = np.std(original_array, axis=1)

variance = np.var(original_array, axis=1)

print("Original 2D Array:")
print(original_array)

print("\nMean along the second axis (axis 1):")
print(mean_values)

print("\nStandard Deviation along the second axis (axis 1):")
print(std_deviation)

print("\nVariance along the second axis (axis 1):")
print(variance)
