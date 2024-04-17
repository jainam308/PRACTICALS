import numpy as np
original_array = np.array([1, 3, 7, 12, 19])
differences = np.diff(original_array)   
print("Original Array:")
print(original_array)

print("\nDifferences between neighboring elements:")
print(differences)