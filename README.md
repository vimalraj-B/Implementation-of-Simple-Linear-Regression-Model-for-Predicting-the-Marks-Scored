# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph.
6.Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: VIMALRAJ B
RegisterNumber:  212224230304
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
df=pd.read_csv('/content/student_scores.csv')

df.head()

df.tail()

X=df.iloc[:,:1].values
X

Y=df.iloc[:,1].values
Y

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression() # Create an instance of the LinearRegression class
regressor.fit(X_train, Y_train)
Y_pred = regressor.predict(X_test)

Y_pred

Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_test,Y_test,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)
```









## Output:
```
DATASET:
```
![Screenshot 2025-04-29 084028](https://github.com/user-attachments/assets/1b371170-f610-409b-94db-0baf6784d37f)

```
HEAD VALUES:
```
![Screenshot 2025-04-29 084159](https://github.com/user-attachments/assets/121bec67-2615-4428-b227-b1b0d5f56246)
```
TAIL VALUES:
```
![Screenshot 2025-04-29 084257](https://github.com/user-attachments/assets/772d13dc-4d8d-4581-ad87-c74dcecaeaa5)
```
X AND Y VALUES:
```
![Screenshot 2025-04-29 084410](https://github.com/user-attachments/assets/528bafc7-efbe-4f4b-87ae-2ed5435f9a1b)
![Screenshot 2025-04-29 084443](https://github.com/user-attachments/assets/c17742d3-0240-4ae2-8fa3-4476488f3e85)
```
PREDICTION VALUES OF X AND Y:
```
![Screenshot 2025-04-29 084600](https://github.com/user-attachments/assets/1d02585a-3e5a-4143-a590-b178b88c3466)
![Screenshot 2025-04-29 084626](https://github.com/user-attachments/assets/3982ac3c-da87-4882-905a-539fa9c945d3)
```
MSE,MAE and RMSE:
```
![Screenshot 2025-04-29 084721](https://github.com/user-attachments/assets/0a4c37fc-39ed-4044-ac84-a889331bef05)
```
TRAINING SET:
```
![Screenshot 2025-04-29 084812](https://github.com/user-attachments/assets/0614f085-0707-4271-ab2f-02eb276608cd)
```
TESTING SET:
```
![Screenshot 2025-04-29 084845](https://github.com/user-attachments/assets/21138c8c-fd08-421e-b741-bac4787a5a99)



```
RESULT:
```
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
