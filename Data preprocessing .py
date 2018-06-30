import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Data preprocessing 


''' Import dataset'''
path ='H:\ML PDF\Data ML\Machine Learning A-Z Template Folder\Part 0 - Welcome to Machine Learning A-Z\Data.csv'
dataset = pd.read_csv(path)
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,3].values


''' remove missing value or clearning data'''
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer = imputer.fit(X[:,1:3])
X[:,1:3] =imputer.transform(X[:,1:3])


'''Encoding categorical data for X '''
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:,0] = labelencoder_X.fit_transform(X[:,0])



'''Encoding Categorical Data for X'''
from sklearn.preprocessing import OneHotEncoder
oneHotEncoder = OneHotEncoder(categorical_features=[0])
X = oneHotEncoder.fit_transform(X).toarray()



'''Encoding categorical data for Y '''
labelencoder_Y = LabelEncoder()
y = labelencoder_Y.fit_transform(Y)




'''Spliting the dataset makeing machine learning Model'''
from sklearn.cross_validation import train_test_split
X_tain , X_test , Y_train, Y_test = train_test_split(X,y,test_size = 0.2)




'''Feature Scaling'''''
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_tain)
X_test =  sc_X.transform(X_test)



'''Final Part of Data preprocessing '''






























