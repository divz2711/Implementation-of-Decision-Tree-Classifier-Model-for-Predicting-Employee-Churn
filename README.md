# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Use standard Python libraries.
2. Set variables for dataset values (X and y).
3. Import Linear Regression from scikit-learn.
4. Predict values.
5. Calculate accuracy metrics.
6. Obtain a graph (visualization of the regression line).

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Divya S
RegisterNumber: 212221040042
*/

import pandas as pd
data=pd.read_csv("/content/Employee[1].csv")

data.head()

data.info()

data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

data["Departments "]=le.fit_transform(data["Departments "])
data.head()

data.info()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

data["Departments "]=le.fit_transform(data["Departments "])
data.head()

data.info()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)

x_train.shape
x_test.shape
y_train.shape
y_test.shape

from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

print(y_pred)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])


```

## Output:
## Initial data set:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/b533c8c0-80a4-4e4c-a0bb-b00dad4004d2)


## Data info:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/de685908-7222-4a8b-ab10-6f68ee6521b8)

## Optimization of null values:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/083e13e3-3821-4cae-9f3d-4bed50b152b6)

## Assignment of x value:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/5138b0f1-d1a7-4509-8b08-fb0e909b6208)

## Assignment of y value:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/56dd0dfd-0de5-4d65-8a71-e59603d33786)


## Converting string literals to numerical values using label encoder:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/4972409e-a02a-45ac-b108-3721a241fa56)

## Accuracy:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/3f832992-2634-4fb7-b880-fd531bc464f3)

## Prediction:
![image](https://github.com/divz2711/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/121245222/08850339-5560-41dd-b5ea-74d4b5c0e699)


## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
