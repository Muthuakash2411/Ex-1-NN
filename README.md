<H3>ENTER YOUR NAME : MUTHUAKASH M</H3> 
<H3>ENTER YOUR REGISTER NO : 212225230194</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 03/08/2026 </H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:

````python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv("Churn_Modelling.csv", index_col="RowNumber")


print("DATASET:")
print(data.head())

print("\nMISSING VALUES:")
print(data.isnull().sum())

print("\nDUPLICATE VALUES:")
print(data.duplicated().sum())

data = data.drop(['Surname', 'Geography', 'Gender'], axis=1)

X = data.iloc[:, :-1].values
Y = data.iloc[:, -1].values.reshape(-1,1)

x_scaler = MinMaxScaler()
X = x_scaler.fit_transform(X)

y_scaler = MinMaxScaler()
Y = y_scaler.fit_transform(Y)

print("\nX VALUES:")
print(X)

print("\nY VALUES:")
print(Y)

Xtrain, Xtest, Ytrain, Ytest = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

print("\nX TRAIN:")
print(Xtrain)

print("\nX TEST:")
print(Xtest)

print("\nY TRAIN:")
print(Ytrain)

print("\nY TEST:")
print(Ytest)
````

## OUTPUT:
<img width="1090" height="567" alt="image" src="https://github.com/user-attachments/assets/6212bd70-238e-45a1-8e0c-33cae04f3817" />

<img width="1096" height="663" alt="image" src="https://github.com/user-attachments/assets/a5467552-adca-42de-9f09-2b961bb3eb8d" />

<img width="1017" height="666" alt="image" src="https://github.com/user-attachments/assets/06e5ba8b-aeaf-4545-9e8e-516735ec1da5" />

<img width="1062" height="428" alt="image" src="https://github.com/user-attachments/assets/9ecb2762-e49d-4e36-824e-12aba37b71e7" />


## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


