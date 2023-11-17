# Ex.No.1---Data-Preprocessing
## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

##REQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

Kaggle :
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

Data Preprocessing:

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

Need of Data Preprocessing :

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
Importing the libraries
Importing the dataset
Taking care of missing data
Encoding categorical data
Normalizing the data
Splitting the data into test and train

## PROGRAM:
```
name: kotha vamsi
reg No: 212222040081
```
```
#importing libraries
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

#Reading the dataset
df=pd.read_csv("/content/Churn_Modelling.csv", index_col="RowNumber")
df

#Dropping the unwanted Columns
df.drop(['CustomerId'],axis=1,inplace=True)
df.drop(['Surname'],axis=1,inplace=True)
df.drop('Age',axis=1,inplace=True)
df.drop('Geography',axis=1,inplace=True)
df.drop('Gender',axis=1,inplace=True)
df

#Checking for null values
df.isnull().sum()

#Checking for duplicate values
df.duplicated()

#Describing the dataset
df.describe()

#Scaling the dataset
scaler=StandardScaler()
df1=pd.DataFrame(scaler.fit_transform(df))
df1

#Allocating X and Y attributes
x=df1.iloc[:,:-1].values
x
y=df1.iloc[:,-1].values
y

#Splitting the data into training and testing dataset
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)
print(len(x_train))
print(x_test)
print(len(x_test))
```
## OUTPUT:
The dataset
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/ea1fb4af-2397-4d9e-83f4-c9f34d49b6f8)

Dropping unwanted features
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/d73df7b4-4f94-4176-b6e9-76ae22fa7f79)

Checking for null values
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/6d530c0f-7cd6-4f14-8a42-bb06ea053564)

Checking for duplication
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/b52456d4-d59c-406c-b802-abb4cb82e49c)

Describing the dataset
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/f74c5480-81bc-4c67-8d7b-92635dc7d226)

Scaling the values
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/b3908b2d-9ad0-424e-b5ff-af524c1c5481)

X Features
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/3f71bf75-9f79-4403-9e0b-86c3b374150a)

Y Features
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/3503facb-5729-4564-8336-b431401df918)

Splitting the training and testing dataset
![image](https://github.com/kothavamsi1/Ex.No.1---Data-Preprocessing/assets/119828582/4fe4c3d5-c1cc-47f6-88ec-32e17aa7eb87)


RESULT

Thus we have successfully performed Data preprocessing in a data set downloaded from Kaggle















