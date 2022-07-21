import streamlit as st
import pandas as pd
# from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from tabulate import tabulate
import pandas as pd
import json
from prettytable import PrettyTable


st.write("""
# Simple Iris Flower Prediction App
This app predicts the **Iris flower** type!
""")

st.sidebar.header('User Input Parameters')

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4)
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4)
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#iris = datasets.load_iris()
#X = iris.data

iris = pd.read_csv('https://raw.githubusercontent.com/Sam-hw/Iris-Project/main/IRIS.csv')
X = iris.drop('species',axis = 1)
Y = iris['species']


clf = RandomForestClassifier()
clf.fit(X, Y)

prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(['Iris-setosa','Iris-versicolor','Iris-virginica'])

value = ['''
       {
           "_id": {
               "$Col1": "XXXXXXX2443"
           },
           "col2": false,
           "col3": "359335050111111",
           "startedAt": {
               "$date": 1633309625000
           },
           "endedAt": {
               "$date": 1633310213000
           },
           "col4": "YYYYYYYYYYYYYYYYYY",
           "created_at": {
               "$date": 1633310846935
           },
           "updated_at": {
               "$date": 1633310846935
           },
           "__v": 0
       }''']

dictionary = json.loads(value[0])
headers = []
values = []
for key in dictionary:
    head = key
    value = ""
    if type(dictionary[key]) == type({}):
        for key2 in dictionary[key]:
            head += "/" + key2
            value = dictionary[key][key2]
            headers.append(head)
            values.append(value)

    else:
        value = dictionary[key]
        headers.append(head)
        values.append(value)

print(headers)
print(values)
myTable = PrettyTable(headers)

myTable.add_row(values)
print(myTable)




st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
