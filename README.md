# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1.Start

step 2.Import the required packages.

step 3.Import the dataset to operate on.

step 4.Split the dataset.

step 5.Predict the required output.

step 6.Stop. 

## Program:
```

/*
Program to implement the SVM For Spam Mail Detection..
Developed by: Shanmuga priya
RegisterNumber:  212222040153
*/

import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')
data.head()
data.info()
data.isnull().sum()
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

```

## Output:
![image](https://github.com/shanmugapriyatharani/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393427/f24bfa97-cb2a-4e71-ab02-d280bf4a7812)

![image](https://github.com/shanmugapriyatharani/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393427/5c347136-7f95-426f-b300-9f63ccf11596)

![image](https://github.com/shanmugapriyatharani/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393427/a4377309-e72f-448f-877f-cb44e4178c08)

![image](https://github.com/shanmugapriyatharani/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393427/9b66512f-0b73-4c35-9448-5bf5a48f2f64)

![image](https://github.com/shanmugapriyatharani/Implementation-of-SVM-For-Spam-Mail-Detection/assets/119393427/8a4df175-029d-470a-b2b8-1dff3b418507)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
