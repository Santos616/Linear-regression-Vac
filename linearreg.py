import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

#Loading our housing dataset
#We will load our data on house sales in King County to predict house prices using simple (one input) linear regression
dataset = pd.read_csv('datasets/housing.csv')
Y = dataset[['price']]
X = dataset.drop(['price', 'id', 'date'],  axis=1)
#Data_Analysis
#Basic data discovery and analysis with info, describe and head
#using pandas .info() we see we have 18 columns and 21613 records. Pretty much all the features given are already in numeric format.

X.info()
columns = X.columns
columns
X.head()

X.describe()

dataset = dataset.drop(['id', 'date'], axis=1)
dataset.corr(method='pearson')

plt.subplots(figsize=(10,8))
sns.heatmap(dataset.corr())



#statsmodel package can also give us some great insight and summary statistics including p-value
#The statsmodel can actually perform the regression modeling for us , but here I am mainly using it to help determine which variable I should focus on for my Simple Linear Regression (one independent variable) 
#and get a feel of which values are statistically significant. There are techniques when dealing with Multiple Linear Regression (many variable)
# to narrow down to the most significant features/variables usiung Step Wise Regression which include techniques such as Forward Selection and Backward Elimination.



import statsmodels.api as sml
from statsmodels import tools

X_new = tools.add_constant(X)

regressor_OLS = sml.OLS(endog = Y,exog =  X_new).fit()

regressor_OLS.summary()

