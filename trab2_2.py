import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import pandas as pd

def calculate(X,y):
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 82)

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    from sklearn.naive_bayes import GaussianNB
    nvclassifier = GaussianNB()
    nvclassifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = nvclassifier.predict(X_test)
    print(y_pred)

    #lets see the actual and predicted value side by side
    y_compare = np.vstack((y_test,y_pred)).T
    #actual value on the left side and predicted value on the right hand side
    #printing the top 5 values
    y_compare[:5,:]

    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    #finding accuracy from the confusion matrix.
    a = cm.shape
    corrPred = 0
    falsePred = 0

    for row in range(a[0]):
        for c in range(a[1]):
            if row == c:
                corrPred +=cm[row,c]
            else:
                falsePred += cm[row,c]
    print('Correct predictions: ', corrPred)
    print('False predictions', falsePred)
    print ('\n\nAccuracy of the Naive Bayes Clasification is: ', corrPred/(cm.sum()))


# Importing the dataset
excel = pd.ExcelFile('databases/iris_dataset.xls')

dataset = excel.parse(0)
#looking at the first 5 values of the dataset
# dataset.head()

group1=dataset[0:29]
group2=dataset[30:59]
group3=dataset[60:89]
group4=dataset[90:119]

test_group=dataset[120:149]

#X = dataset.iloc[:,:4].values
X = group1.iloc[:,:4].values
#y = dataset['species'].values
y = group1['species'].values
print(X)
calculate(X,y)

X = group2.iloc[:,:4].values
y = group2['species'].values
calculate(X,y)

X = group3.iloc[:,:4].values
y = group3['species'].values
calculate(X,y)

X = group4.iloc[:,:4].values
y = group4['species'].values
calculate(X,y)