import numpy as np
data = np.ndarray((2,3))
print(data)
print(data.shape)
print(data.dtype)

# initialize ndarrays
data1 = [1, 7.5, 2, 42, 13, -3, 0]
arr1 = np.array(data1)
print(arr1)

data2 = [[1,3,5],[2,4,6]]
arr2 = np.array(data2)
print(arr2)
print(arr2.ndim) # 2
print(arr2.shape) # (2, 3)

# initialize 0s and 1s
arr3 = np.zeros((4,4))
print(arr3)

arr4 = np.ones((4,4))
print(arr4)

# empty creates array with random initial values
arr5 = np.empty((2, 3, 2))
print(arr5)


# arange is an array-valued version of the 
# built-in Python range function
arr6 = np.arange(10)
print(arr6)

# set the data type
arr7 = np.array([1, 2, 3.0], dtype=np.int64)
print(arr7)

# convert data type
arr8 = np.array([1.5, 2.3, 3.1])
arr8 = arr8.astype(np.int64)  # create a new array and map and copy 
print(arr8)

# vectorization
arr = np.array([[1., 2., 3.], [4., 5., 6.]])
print(arr)
arr = arr * arr
print(arr)
arr = 1 / arr
print(arr)


# indexing and slicing 
arr3d = np.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
old_values = arr3d[0].copy()
print(old_values)
arr3d[0] = 42
print(arr3d)
arr3d[0] = old_values
print(arr3d)

arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(arr[:, :1]) # print left mode column
print(arr[:2, 1:]) # print columns starting from second column of 
                   # first two rows

# boolean index
# generate a list of names and a dataset
# each name coresponds to a row in the dataset
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
data = np.random.rand(7, 4)
print(data)
table = np.concatenate((names, data.T),axis=0)
print(table)
boolean_arr = np.array([name == 'Bob' for name in names if name in names]) # list comprehension
print(boolean_arr)