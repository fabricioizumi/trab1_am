from sklearn import linear_model,preprocessing

import pandas as pd

#Load database
f=pd.ExcelFile('../hospital.xls')

#parse first worksheet
data=f.parse(0)

#Load group 1
g1=data[0:20]

#Transform string labels 
le1=preprocessing.LabelEncoder()
le_sex=le1.fit(g1['Sex'])

le2=preprocessing.LabelEncoder();
le_smoker=le2.fit(g1['Smoker'])

#set input variables
x=g1[['Age','Weight']]

x['Sex'] = le_sex.transform(g1['Sex'])
x['Smoker'] = le_smoker.transform(g1['Smoker'])

#set real values
y=g1['BloodPressure_1']

lm=linear_model.LinearRegression()

lm.fit(x,y) #load datas

g1_pred=lm.predict(x) # predict

print(g1)
print(g1_pred);
print(y)

print("Score: " + str(lm.score(x,y)))
#print("Coef: " + lm.coef_)

