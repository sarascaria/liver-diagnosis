import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import pickle

df = pd.read_csv('Training.csv')

df.head()

df=df.dropna()

x=df.drop('Disease', axis=1)

x.head()

y=df['Disease']

print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.10, random_state=42)

model= MultinomialNB()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

y_pred

a= accuracy_score(y_test, y_pred)*100
print(a)
df2 = pd.read_csv('Testing.csv')

type(df2.drop('Disease', axis=1))

model.predict(df2.drop('Disease', axis=1))

df2['Disease']

# Saving model to disk
pickle.dump(regressor, open('m.pkl','wb'))
# Load the model
model = pickle.load(open('model.pkl','rb'))
