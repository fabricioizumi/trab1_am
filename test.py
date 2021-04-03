from sklearn import linear_model,preprocessing

import pandas as pd

f=pd.ExcelFile('../hospital.xls')

data=f.parse(0)

g1=data[0:20]

le1=preprocessing.LabelEncoder()
le_sex=le1.fit(g1['Sex'])

le2=preprocessing.LabelEncoder();
le_smoker=le2.fit(g1['Smoker'])

x=g1[['Age','Weight']]

x['Sex'] = le_sex.transform(g1['Sex'])
x['Smoker'] = le_smoker.transform(g1['Smoker'])

y=g1['BloodPressure_1']

lm=linear_model.LinearRegression()
lm.fit(x,y)

g1_pred=lm.predict(x)

print(g1_pred);
print(y)

