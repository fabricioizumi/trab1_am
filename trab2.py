from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score
import numpy as np
import pandas as pd

f=pd.ExcelFile('databases/iris_dataset.xls')

data=f.parse(0)

data2=data.sample(frac=1)
#print(data2)

data2["species"]=np.where(data2["species"]=="setosa",'st',
                            np.where(data2['species'] == 'versicolor','vs', 'vg'))



group1=data2[0:29]
group2=data2[30:59]
group3=data2[60:89]
group4=data2[90:119]

df=data2
print(df)
group4=group4[[
    "meas1",
    "meas2",
    "meas3",
    "meas4",
]].dropna(axis=0, how='any')

test_group=data2[120:149]

X = df.iloc[:,:-1]
y = df.iloc[:,-1]


gnb = GaussianNB()
used_features =[
    "meas1",
    "meas2",
    "meas3",
    "meas4"
]

#Implementing cross validation
 
k = 5
kf = KFold(n_splits=k, random_state=None)

acc_score = []

for train_index , test_index in kf.split(X):
    X_train , X_test = X.iloc[train_index,:],X.iloc[test_index,:]    
    y_train , y_test = y[train_index] , y[test_index]
     
    # Train classifier
    gnb.fit(
        X_train,
        y_train
    )

    pred_values = gnb.predict(X_test)
     
    acc = accuracy_score(pred_values , y_test)
    acc_score.append(acc)
     
avg_acc_score = sum(acc_score)/k
 
print('accuracy of each fold - {}'.format(acc_score))
print('Avg accuracy : {}'.format(avg_acc_score))


# y_pred = gnb.predict(test_group[used_features])

# #accuracy = accuracy_score(test_group,y_pred)
# precision =precision_score(test_group, y_pred,average='micro')
# f1 = f1_score(test_group,y_pred,average='micro')

# print('Confusion matrix for Naive Bayes\n',cm)
# #print('accuracy_Naive Bayes: %.3f' %accuracy)
# print('precision_Naive Bayes: %.3f' %precision)
# print('f1-score_Naive Bayes : %.3f' %f1)