# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.


## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Vaishnavi S.A
RegisterNumber: 212223220119 
*/
```
import pandas as pd
df=pd.read_csv('Employee.csv')
df
df.head()
df.info()
df.isnull().sum()
df["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['salary']=le.fit_transform(df['salary'])
df.head()
df["Departments "].value_counts()
df['Departments ']=le.fit_transform(df['Departments '])
df.head()
x=df[['satisfaction_level','last_evaluation','number_project','average_montly_hours','time_spend_company','Work_accident','promotion_last_5years','Departments ','salary']]
x.head()
x.info()
y=df['left']
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
x_train.shape
x_test.shape
y_train.shape
y_test.shape
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion='entropy')
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,8,9,200,6,0,1,1]])
'''


## Output:
![decision tree classifier model](sam.png)

![281462537-16a91699-c138-436a-a728-ff856cfcce0a](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/548d9e21-32fa-4001-bc23-acf39d377ee9)


![281462964-b201d728-2f3e-474e-bfdb-5ea2c3733f58](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/d970359e-d51c-47c3-a6d8-34e5745aa7a2)

![281463018-c878e2d0-4bdc-4747-bf4b-87943ec7b68d](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/a1c7bda7-1e19-4bb5-a752-b70dca8e41e7)

![281463194-1d435e48-d60a-4e1c-8957-06c8f4e033c5](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/7e49d3b4-ce11-46c4-9111-88c2a719aa17)

![281463466-2be85711-67d7-4c61-aae7-e937ab760372](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/4d1b8316-0f15-44d6-9a36-7125f685df05)


![281464022-ed9397d2-bd24-4446-adc1-555fbed64415](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/d2fe1906-ac40-44f6-ba12-bf8bfa10b213)

![281464166-26bdd0cd-b033-42f8-ae21-9f8c3bf25d5b](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/319687f1-e9c7-4a9e-97f5-6543f8bb7d8d)

![281464612-f4c7cd07-87c1-4bb8-8b55-a55e9a50537e](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/371759c3-3bd8-492e-99c0-fd97b7e28d1d)

![281464682-83f12c43-d21b-4e98-9112-46ffc2b5fc95](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/59ba1e49-892a-4f5f-8dc7-5018722ebdae)

![281464725-590de63f-46e7-4e29-bb37-fac051e4742b](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/5c7268ca-aa4b-46c3-a401-1f6a978e2798)

![281464808-b45ecc21-075d-4190-9951-6099312b0a58](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/bdfa48bd-6b52-4c21-8031-164dff479708)

![281465014-be2e03cf-aec8-436f-9f73-658c5e6e5127](https://github.com/vaishnavishaji/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/151444759/e15a1459-3a84-4493-a5e8-36079fdeb229)















## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
