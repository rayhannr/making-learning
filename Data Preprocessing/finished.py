# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values #kolom 1-3 (feature, independent)
y = dataset.iloc[:, 3].values #kolom 4 (result, dependent)

# Taking care of missing data
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0)
missingvalues = missingvalues.fit(X[:, 1:3])
X[:, 1:3] = missingvalues.transform(X[:, 1:3])

# Encoding categorical data (country and purchase)
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
#args ketiga itu maksudnya yang ditransform itu kolom pertama (index 0)
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough') #kolom negara dibikin tiga kolom france, germany, spain
X = np.array(ct.fit_transform(X), dtype=np.float)
#di X sekarang 0=france, 1=germany, 2=spain

# Encoding the Dependent Variable
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y) #no=0, yes=1

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
#train_test_split params : awal itu untuk array apa aja, kedua: testSize (0.2 = 20%, good practice)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
#feature scaling itu biar fitur2nya punya scale yang sama sehingga tidak ada yang mendominasi
#add dua cara feature scaling, standardization dan normalization.
#standardization: xstand = (x-mean(x))/standar deviasi x
#normalization: xnorm = (x-min(x))/(max(x) - min(x))
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
#kalo training set harus difit dulu baru ditransform
X_train = sc_X.fit_transform(X_train)
#kalo test set langsung ditransform karena scX udah difit di train set
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))