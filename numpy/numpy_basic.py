import numpy as np
import random as rand

data = np.ndarray((2, 3))
print(data)
print(data.shape)
print(data.dtype)

# initialize ndarrays
data1 = [1, 7.5, 2, 42, 13, -3, 0]
arr1 = np.array(data1)
print(arr1)

data2 = [[1, 3, 5], [2, 4, 6]]
arr2 = np.array(data2)
print(arr2)
print(arr2.ndim)  # 2
print(arr2.shape)  # (2, 3)

# initialize 0s and 1s
arr3 = np.zeros((4, 4))
print(arr3)

arr4 = np.ones((4, 4))
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
print(arr[:, :1])  # print left mode column
print(arr[:2, 1:])  # print columns starting from second column of
# first two rows

# boolean index
# generate a list of names and a dataset
# each name coresponds to a row in the dataset
names = np.array(['Bob', 'Joe', 'Will', 'Bob', 'Will', 'Joe', 'Joe'])
# generate random data
data = np.random.rand(7, 4)
print(data)
# transpose names to column
names = np.array([names])
names = names.T
# print(names.ndim)
# concatenate names with data into a table
table = np.append(names, data, axis=1)
print(table)
# find 'Bob' in the table
boolean_arr = np.array([name == 'Bob' for name in names if name in names])  # list comprehension
print(boolean_arr)

# Fancy Indexing
arr = np.empty((8, 4))
for i in range(8):
    arr[i] = i
print(arr)
print(arr[[4, 3, 0, 6]])
print(arr[[-1, -2, -3]])

arr = np.arange(32).reshape((8, 4))
print(arr)
print(arr[[0, 1, 2, 3], [0, 1, 2 ,3]]) # print corresponding elements in the format of array
print(arr[[0, 1, 2, 3]][:, [3, 2, 1, 0]]) # get rows and then processed by the index in second parenthesis

# Transposing and Swapping
arr = np.arange(15).reshape((3, 5))
arr = arr.T
print(arr)

# ufunc (universal functions)
arr1 = np.random.rand(8)
arr2 = np.random.rand(8)
print(np.maximum(arr1, arr2))  # element-wise maximum

# points = np.arange(-5, 5, 0.01) # low, high, step
# print(points)
# # The np.meshgrid function takes two 1D arrays and pro- duces two 2D
# # matrices corresponding to all pairs of (x, y) in the two arrays:
# xs, ys = np.meshgrid(points, points)
# # print(xs)
# import matplotlib.pyplot as plt
# z = np.sqrt(xs**2 + ys ** 2)
# print('\n\n\n\n\n')
# print(z)

# plt.imshow(z, cmap=plt.cm.gray)
# plt.colorbar()
# plt.title("Image plot of $\sqrt{x^2 + y^2}$ for a grid of values")
# plt.show()


# np.where
arr = np.random.randint(-5, 5, (4, 4))  # low, high, size
print(arr)
# arr = np.where(arr > 0, 2, -2)
# print(arr)

arr = np.where(arr > 0, 2, arr) # only set positive values to 2
print(arr)

# Mathematical and Statistical Methods
arr = np.random.rand(5, 5)
print(np.mean(arr))
print(np.sum(arr))

# Axes are defined for arrays with more than one dimension. A 2-dimensional array has two corresponding axes:
# the first running vertically downwards across rows (axis 0), and the second running horizontally
# across columns (axis 1).
arr = np.random.rand(5, 4)
print(arr)
print(np.sum(arr, axis=0))  # calculate sum on each column
print(arr.sum(1)) # calculate sum on axis = 1 which is horizontal (each row)

# np sort
arr = np.random.rand(1, 5)
print('\n')
print(np.sort(arr))

arr = np.array([1, 2, 3, 1, 2, 5, 6, 7, 2, 1])
print(np.unique(arr))


# Linear Algebra
# dot operation
x = np.array([[1., 2., 3.], [4., 5., 6.]])
y = np.array([[1, 4], [2, 5], [3, 6]])
print(np.dot(x, y))


# inverse
from numpy.linalg import inv
X = np.random.randint(-1, 1, (5, 5))
mat = X.T.dot(X)
# print(inv(mat))
print(mat.dot(inv(mat)))

# Random Number Generation
# np.random.rand # samples from a uniform distribution
# np.random.randn # Draw samples from a normal distribution with mean 0 and standard deviation 1 (MATLAB-like interface)
