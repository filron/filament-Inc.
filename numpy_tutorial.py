'''
1-D Array -> Vector
2-D Array -> Matrix
3-D Array -> Tensor
'''


from numpy import array
from numpy import ones
from numpy import zeros
from numpy import empty
from numpy import vstack
from numpy import hstack
from numpy import arange
from numpy import sort

#SUBPACKAGES
#doc:indexing and typical documentation
#lib:Basic function used by sub packages
#random:core random tools
#linalg:core Linear Algebra tools
#fft:core fft routines
#polynomial:polynomial tools
#testing:testing tools
#f2py:fortran to python interface generator
#distutils:support for fortran compilers

#__Version__:show numpy build version
#show_config:show build configuration for numpy
#dual:overwrite certain function with high-performance scipy packages
#matlib:make everything matrices


'''concepts

Broadcasting
Flattening
Newaxis
Reshaping
Dimentional shuffling





'''




"""DATATYPES
complex128
complex64
float16
float32
float64
int8
int16
int32
int64
timedelta64
uint8
uint16
uint32
uint64
"""

data=[1.0,2.0,3.0,4.0]
arr_og=np.array(data)
print(arr_og)

#Display the dimentions of the array
arr_og.shape

#Display Datatype of the array
arr_og.dtype

#Display the number of row in an array
arr_og.ndim

#Display the length of an array
len(arr_og)

#Array filled with 1's of specified size (Default data type = float64)
arr_ones=np.ones([4] dtype="float64")
print(arr_ones)

#Array filled with 0's of specified size (Default data type = float64)
arr_zeros=np.zeros([4], dtype="float64")
print(arr_zeros)

#To create an empty array of 3 row and 5 col
arr_empty=np.empty([3][5])
print(arr_empty)

#To combine arrays
arr_1=np.array([1,2,3])
arr_2=np.array([4,5,6])

#Vertical stack
arr_3=np.vstack((arr_1,arr_2))
print(arr_3)

#Horizontal stack
arr_3=np.hstack((arr_1,arr_2))
print(arr_3)

#To Create a one dimentional array (List)
data_1d=[1,2,3]
arr_1d=np.array(data_1d, dtype="int64")

#To create a 2 dimentional array (List->List)
data_2d=[[1,2,3],[4,5,6],[7,8,9]]
arr_2d=np.array(data_2d, dtype="int64")

#To create a 3 dimentional array (List->List->List)
data_3d=[[[1,2,3],[4,5,6],[7,8,9]],[[11,12,13],[14,15,16],[17,18,19]],[[21,22,23],[24,25,26],[27,28,29]]]
arr_3d=np.array(data_3d, dtype="int64")

#To create an array using arange function
arr_og=np.arange(0,10,2)
print(arr_og)
#arg1=start number
#arr2=end number(Excluded)
#arr3=step size

#To sort elements of array
arr_unsort=[2,3,1,4,5,7,6,8,9]
print("Array before sorting:",arr_unsort)
arr_sort=arr_unsort.sort()
print("Array after sorting:",arr_sort)

#To sort elements of array in row manner
arr_sort_row=arr_unsort.sort(axis=1)

#To sort elements of array in column manner
arr_sort_col=arr_unsort.sort(axis=0)

#To sort selective elements on an array
arr_sort_selected_index=np.sort([1,2,3])

#To get the orignal index of elements of array after sorting elements
arr_index=np.argsort(arr)

#To calaculate the mean of the array
print("Mean of the array:")
arr_mean=arr_og.mean()

#To calculate the variance of the array
print("Variance of the array:")
arr_variance=arr_og.var()

#To calculate the standard deviation of the array
print("Standard deviation of the array:")
arr_sd=arr_og.std()

#To create a linear space by dividing range into equal specified parts
arr_og=np.linspace(start=0,10,num=5,retstep=True,endpoint=False)
#arg1=start number
#arg2=end number
#num=number of parts to divide
#retstep=returns the step size (ie diffrence between two elements)
#endpoint=includes/excludes the endpoint into division

#To create an identity matrix of specified dimentions
row=3
col=2
arr_identity=np.eye(row, col)

#To assign values to the diagonal elements of a matrix
arr_diagonal=[1,2,3,4]
np.diag(arr_diagonal)


#To get a subarray of an array
arr_sliced=arr_og[0:10:2]

#To reverse the elements of an array
arr_forward=[0,1,2,3,4,5,6,7,8,9]
arr_reversed=arr_forward[::-1]

#To share memory (Referenced memory) of any subarray
arr_parent=arange(0,10,1)
arr_child=arange(0,10,2)
np.shares_memory(arr_parent, arr_child)
#change in value of child array index will change the value in parent array too

#To multiply 2 arrays
arr_1.dot(array_2)
arr_1*arr_2

#To compare 2 arrays
np.array_equal(arr_1,arr_2)

#To compare elements of 2 arrays
arr_1==arr_2

#To get an sin angle of array elements
np.sin(arr)

#To get a logarithmic value of elements of array
np.log(arr)

#To get a exponential value of elements of array
np.exp(arr)

#To add all the elements of an array
np.sum(arr)

#To add column elements of an array
np.sum(arr, axis=0)

#To add row elements of an array
np.sum(arr, axis=1)

#To get the minimum valued element
arr.min()

#To get the maximum valued element
arr.max()

#To get the index of minimum valued element
arr.argmin()

#To get the index of maximum valued element
arr.argmax()

#return true if all elements are true
np.all()

#return true if any elements are true
np.any()


#To load data into numpy
np.loadtxt("abc.txt", delimiter=',', unpack=True, converters={0:})

#To implement broadcasting in numpy
np.tile(arr,(col,row))

#To transpose a array
arr.T

#To implement flattening of a 2D array into a 1D array
np.ravel(arr)
arr.ravel()


#To convert a 1D array into a 2D array
np.newaxis


#To create an array of mXn dimention of constant value
np.full((m,n),value)

#To display the diffrence between elements of an array
np.diff(arr,n)
#n=number of itterations od diffrences

#to divide a scalar
np.divide(x,y)
np.floor_divide(x,y)
np.true_divide(x,y)
np.remainder(x,y)
np.multiply(x,y)



np.c_[]
np.r_[]
np.einsum()
np.tile(a,reps=())
np.logical_and(arg1, arg2)
