import numpy as np

arr1 = np.arange(50)

arr2 = arr1[arr1 % 2 == 0]
print(arr2)

result = np.mean(arr2)
print(result)


  