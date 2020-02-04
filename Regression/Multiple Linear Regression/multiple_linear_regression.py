# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encoding categorical data
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)

# Avoiding the Dummy Variable Trap
X = X[:, 1:] #it means we take all the rows and columns except column with index = 1. kita hapus satu categorical var

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

#building the optimal model using backward elimination
import statsmodels.api as sm
#nambah 1 kolom dengan 50 baris di paling kiri matrix X yang isinya 1 semua. ini buat default value (b0)
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1) #axis=1 berarti nambah kolom. kalo = 0 berarti nambah baris
"""X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #fit the full model with all possible predictors
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]] #2 dihapus karena ketika di summary, dia punya p value tertinggi dan ternyata lebih besar dari 0.05, makanya dihapus sesuai dengan metode backward elimination
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #fit the full model with all possible predictors
regressor_OLS.summary()
X_opt = X[:, [0, 3, 4, 5]] #1 hapus dengan alasan sama kek no 2. ini dilakukan terus sampe gaada independent var yang p valuenya > 0.05
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #fit the full model with all possible predictors
regressor_OLS.summary()
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #fit the full model with all possible predictors
regressor_OLS.summary()
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit() #fit the full model with all possible predictors
regressor_OLS.summary()"""

def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)
regressor.fit(X_train, y_train)
y_pred2 = regressor.predict(X_test)