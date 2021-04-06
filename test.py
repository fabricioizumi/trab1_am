from sklearn import linear_model,preprocessing

import pandas as pd

def calc(data):
    #Transform string labels 
    le1=preprocessing.LabelEncoder()
    le_sex=le1.fit(data['Sex'])

    le2=preprocessing.LabelEncoder()
    le_smoker=le2.fit(data['Smoker'])

    #set input variables
    x=data[['Age','Weight']]

    x['Sex'] = le_sex.transform(data['Sex'])
    x['Smoker'] = le_smoker.transform(data['Smoker'])

    #set real values
    y=data['BloodPressure_1']

    lm=linear_model.LinearRegression()

    lm.fit(x,y) #load datas

    data_pred=lm.predict(x) # predict

    print(data)
    print(data_pred);
    print(y)

    print("Score: " + str(lm.score(x,y)))

#Load database
f=pd.ExcelFile('../hospital.xls')

#parse first worksheet
data=f.parse(0)

#Load group 1
g1=data[0:20]
g2=data[21:40]
g3=data[41:60]
g4=data[61:80]

#print("Coef: " + lm.coef_)
calc(g1)
calc(g2)
calc(g3)
calc(g4)