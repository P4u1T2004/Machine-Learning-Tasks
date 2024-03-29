import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

breast_cancer_dataset = sklearn.datasets.load_breast_cancer()

print(breast_cancer_dataset)

data_frame = pd.DataFrame(breast_cancer_dataset.data, columns = breast_cancer_dataset.feature_names)

data_frame.head

data_frame['label'] = breast_cancer_dataset.target

data_frame.shape

data_frame.info()

x = data_frame.drop(columns='label', axis=1)
y = data_frame['label']

print(x)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.2, random_state=2)

print(x.shape,x_train.shape,x_test.shape)

model= LogisticRegression()

model.fit(x_train,y_train)

x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train,x_train_prediction)

print(training_data_accuracy)

x_test.prediction =model.predict(x_test)
testing_data_accuracy = accuracy_score(y_test, x_test.prediction)

print(testing_data_accuracy)
