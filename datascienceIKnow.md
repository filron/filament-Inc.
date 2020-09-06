# Data Analysis / Data Science / Mahine Learning / Deep Learning

#### Index
1. Modules/Libs
	* Numpy
	* Pandas
	* Matplotlib
	* Sklearn (scikit-learn)
2. Visualization
	* Histogram
	* Scatter plot
		* 2D Scatter plot
		* 3D Scatter plot
	* Pair plot
	* Box and Whisker plot
	* Joint plot
	* 
	* 
	* 
3. Regression
	* Simple Linear Regression
	* Multi Linear Regression
	* Polynomial Regression
	* Logistic Regression
	* Ridge Regression
	* Lasso Regression
	* Ordinary Least Square Regression
4. SVM
	* Non Kernelized SVM
	* Kernelized SVM
5. KNN
6. Trees

*Appendix A: Most asked Questions*

*Appendix B: Facts / Tips / Tricks*

*Appendix C: Mathemathics for Machine Learning*

*REFERENCES*



## Modules/Libs

Numpy
Pandas
Matplotlib
Seaborn
Scikit-Learn
Plotly

pip
dlib
cmake
opencv-python
imutils
itertools
scipy
merkletools
tensorflow
keras
pytorch
gluoncv
mxnet


## Opencv
```python
import cv2
cv2.videoCapture(0).read()
cv2.destroyAllWindows()
```
## Matplotlib

```python
import numpy as np
import matplotlib.pyplot as plt

a=[0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]
b=[180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345, 360]

#plot 1
plt.plot(np.sin(a+b), label='sin(a+b)')
plt.plot((np.sin(a)*np.cos(b))+(np.cos(a)*np.sin(b)))

#plot 2
plt.plot(np.sin(a-b), label='sin(a-b)')
plt.plot((np.sin(a)*np.cos(b))-(np.cos(a)*np.sin(b)))

#plot 3
plt.plot(np.cos(a+b), label='cos(a+b)')
plt.plot((np.cos(a)*np.cos(b))-(np.sin(a)*np.sin(b)))

#plot 4
plt.plot(np.cos(a-b), label='cos(a-b)')
plt.plot((np.cos(a)*np.cos(b))+(np.sin(a)*np.sin(b)))

#plot 5
plt.plot(np.tan(a+b), label='tan(a+b)')
plt.plot((np.tan(a)+np.tan(b))/(1-(tan(a)*np.tan(b))))

#plot 6
plt.plot(np.tan(a-b), label='tan(a-b)')
plt.plot((np.tan(a)-np.tan(b))/(1+(tan(a)*np.tan(b)))

#plot 7
plt.plot(np.sin(a)+np.sin(b), label='sin(a)+sin(b)')
plt.plot(2*(np.sin((a+b)/2)*np.cos((a-b)/2)))

#plot 8
plt.plot(np.sin(a)-np.sin(b), label='sin(a)-sin(b)')
plt.plot(2*(np.cos((a+b)/2)*np.sin((a-b)/2)))

#plot 9
plt.plot(np.cos(a)+np.cos(b), label='cos(a)+cos(b)')
plt.plot(2*(np.cos((a+b)/2)*np.cos((a-b)/2)))

#plot 10
plt.plot(np.cos(a)-np.cos(b), label='cos(a)-cos(b)')
plt.plot(2*(np.sin(a+b)/2)*np.sin((a-b)/2)))

```
## Numpy

```python
import numpy as np
np.mean()
np.median()
np.std()
np.percentile()
np.cbrt()
```

## Bayesian Mathemathics (Statistics)

bayes theorem (probablistic theorem)
conditional probablity
observed probablity
likelihood

monte carlo methods based on bayesian methods

hypothesis
false positive
true positives
false negetives
true negetives


## Regression Analysis
  * Simple Linear Regression
  * Multi Linear Regression
  * polynomial Regression
  * Logistic Regression
  * ridge Regression
  * Lasso Regression
  * Ordinary Learst Square Regression (OLS)

## Toy Dataset

iris dataset
fashion mnist dataset
titanic dataset
pixar_movies dataset
cancer dataset


## Type of Plots
plot
scatter plot (2D and 3D)
box plot (whisker plot)
pair plot
quiver plot
histogram:
	used to plot density of variable
violin plot
joint plot
swarm plot
heat map / plot
polar plot
stream line plot
barb plot
polar scatter plot

## Simple Linear Regression

Y = c+mX

Y : dependent variable
X : independent variable
c : constant (yintercept) ie. point where line cuts the dependent variable (Y)
m : slope of the line (rise/run)
	X2-X1/Y2-Y1

Note:
1. Y is a scalar value of dependent variable
2. X is an Array/Vector of independent variable
3. m is a scalar constant value (slope of a linear model is constant)
4. c is a scalar constant value

## Multi Linear Regression

Assumption of Linear Regression
* Linearity of data
* multivariate normality
* Independence of error
* Lack of multi collinearity


Y = c+m1X1+m2X2+m3X3+...+mnXn

Y : dependent variable
X : independent variable
c : constant (yintercept) ie. point where line cuts the dependent variable (Y)
m : slope of the line (rise/run)
	X2-X1/Y2-Y1


```python
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train, y_train)
ypred=reg.predict(x_test)
plt.scatter(x_train, y_train, color='red')
plt.plot(x_train, reg.predict(x_train), color='blue')
```

Note:
1. do not add all dummy variable in model (this is called Dummy variable trap)
if you have `n` dummy variables, use `n-1` variables in model
i.e. for 9 dummy variables use 9-1 (8) variables
2. unlike Simple Linear Regression where X was an Array/Vector, in Multi Linear Regression, X is a Matrix or 2 or more independent variables and Y is an Array/Vector of dependent variable
1. X is a Matrix of 2 or more independent variables
2. Y is an Array/Vector of dependent variables
3. 

## Polynomial Regression

Y = c+m1X^1+m2X^2+m3X^3+...+mnX^n
# incomplete: to be completed

Note:
1. Polynomial Regression is a special case of Multi Linear Regression

```python
from sklearn.Linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(x_train, y_train)
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x_train)
lin_reg_pol=LinearRegression()
lin_reg_poly.fit(x_poly, y_train)

```

## Support Vector Machines

## Decision Tree and Random Forest Tree

## K Nearest Neighbour

```python
#imports
from sklearn.dataset import load_breast_cancer
from sklearn.neighbors import KNeignborsClassifier
from sklearn.preprocessing import train_test_split
import mglearn

#load dataset
cancer = load_breast_cancer()
print(cancer.DESCR)
print(cancer.feature)
print(cancer.feature_names)
print(cancer.target_names)
print(cancer.data)
print(cancer.data.shape)

#fiting model
mglearn.plot.plot_knn_classification(n_neighbors=3)
x_train, y_train, x_test, y_test = train_test_split(cancer.data, cancer.taget,
	stratify=cancer.target, random_state=42)
knn=KNeignborsClassifier()
knn.fit(x_train, y_train)
knn.score(x_train, y_train)
knn.score(x_test, y_test)
# change the value of n_neighbors and % of train_test_split for better results

```


## Label encoder in scikit-learn

```python
#label encoder
from sklearn.preprocessing import LabelEncoder
labelencoder=LabelEncoder()
X[:,0]=labelencoder.fit_transform(X[:,0])
```

## One Hot Encoder in Scikit-learn

```python
#one hot encoder : 
from sklearn.preprocessing import OneHotEncoder
onehotencoder=OneHotEncoder(categorical_feature=[0])
X = onehorencoder.fit_transform(X).toarray()
```

## Imputer in scikit-learn

```python
#imputer : fixing missing values with mean
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', stratergy='mean', axis=0)
# X[:, 1:3] are the features to apply
imputer=imputer.fit(X[:, 1:3])
X[:, 1:3]=imputer.transform(X[:, 1:3])
```

## train-test-split in scikit-learn

```python
#train test split : spliting data fot training, testing and evaluating model
from sklearn.cross_validation import train_test_split
x_train, y_train, x_test, y_test = train_test_split(X,Y, test_size=0.40, random_state=True)
```

## Feature scaling (Normalization and Standardization)

to put all features values  on same scale so no other feature are domninated by features with large scale

Normalization : putting each observation on a relative scale between the value [0 & 1]
	(value of observation)/(sum of all observations)
	
Standardization : rescaling data so it has no zero (0) mean and unit variance
```python
#standardization
#Scale train features first then test features
from sklearn.preprocessing import StandardScalar
sc_x=StandardScalar()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.transf
```
## *Other
### Normal distribution (Gaussian distribution)
has continous PDF (Probablity Density Function)
# *Appendix A: Questions
* what is data point/observation?
* what are outliers and how to fix them?
* when to use histogram?, box ans whiskers?, pair plot, 
* what is EDA (Exploratory data analysis)?
* how to visualize multi dimentional data?
* how to visualize 2 dimentional data?
* what is IQR (Inter Quartile Range)?
* what is PDF (Probablity Density Function) and PMF (Probablity Mass Function)?
* how to visualize univariate and multi variate data?
* how to load datasets?
* how to handle missing data?
* how to select best feature from model?
* what is correlation?
* how to visualize correlated data?
* how to encode and decode data from one type to another?
* what is one hot encoder?
* what is catergorical data?
* how to handle missing categorical data?
* what is numerical data?
* how to handle missing numerical data?
* how to rescale data for modeling?
* what is data augmentation?
* what is the correct amount of data?
* how is data distributed as training data and testing data?
* what is validation data?
* what are the parameters on which a models performance is validated?
* is accuracy a good performance indicator?
* is precission a good performance indicator?
* is speed a good performance indicator?
* what ETL (Extract, Transform, Load) and where is it used?
* what is feature extraction and data cleaning/
* why is there a need of data cleaning?
* can model automatically clean data and apply algorithms on data?
* which is best model to select for any project?
* what is cross validation?
* what is k fold cross validation?
* why is preprocessing of data important?
* what benefits can we get by doing EDA?
* how to deploy models?
* what is MSE (Mean Square Error)?
* what is cross entropy?
* is cross entropy better than MSE?
* what are the types on ml models?
* what are descriptive models and predective models?
* what are statistical methods/models?
* what is usuall percentage split for train and test data?
* how to estimate the performance of a ml model?
* what is ml modelling life cycle?
* what are the stages of ml modelling lifecycle?
* what is label encodeing?
* what is standardization and normalization? and where are they used?
* what is feature scaling and why is it needed?
* what is least square?
* how to encode categorical data into numerical data?
* what is a predictor, feature, column?
* what is datapoint, row, observation?

# *Appendix B: Others / Tricks / Tips /

* mean and variance are corupted by outliers
* median is not corupted by outliers
* median can only be corupted if half the data points are outliers
* variance is squared difference
* 

# *Appendix C: Mathemathics for Machine Learning

1. Linear Algebra
	* PCA (principal component analysis)
	* SVD (single value decomposition)
	* Eigen decomposition of a matrix
	* LU decomposition
	* QR decomposition
	* Symmetric matrices
	* orthogonalization and orthonormalization of matrix
	* matrix operations
	* projection
	* eigen vectors and eigen values
	* vector spaces and norms
	
2. Calculus
    * Multivariate calculus
	* basic differential and integral calculus
	* partial derivatives
	* vector value functions
	* directional gradient
	* hessian, jacobian, lagrange, laplacian distributions
	* optimization theory
	
3. Statistics and Probablity
    * probablity theory
	* combinatorics
	* probablity rules and axioms
	* bayes theorrem
	* random variables
	* variance and expectations
	* conditional and observed probablities
	* conditional and disjoint distributions
	* standard deviation
	* bernoullis, binomial, multinomial, uniform, gaussian probablity distribution
	* moment generating functions
	* MLE (maximum likelihood functions)
	* MAP (maximum aposteriori estimation)
	* priori and posterior
	* sampling methods
	
4. Algorithm and complex optimization
	* binary tree
	* hashing techniques
	* heap
	* stack
	* dynamic programming, divide and conquer, greedy algorithms
	* randomized and sublinear algorithms
	* graph, gradient, stochastic descents
	* primal dual methods
  
# *References
	
  * [towards data science (the mathemathics of machine learning) Wale Akinfaderin](https://towardsdatascience.com/the-mathematics-of-machine-learning-894f046c568)
	* 






