import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, ensemble

sns.set_style('whitegrid')

data_train = pd.read_csv('D:/Github/titanic_sklearn/resources/titanic_train.csv')
data_test = pd.read_csv('D:/Github/titanic_sklearn/resources/titanic_test.csv')

#selection of necessary information
data_train = data_train.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)
# fill in the gaps
data_train['Embarked'] = data_train['Embarked'].fillna('S')
# data cleaning
data_train['Age'].fillna(data_train['Age'].median(), inplace=True)
data_train['Cabin'].fillna('XXX', inplace=True)

data_transfer = data_test.drop(['Name', 'Ticket', 'Fare'], axis=1)

#selection of necessary information
data_test = data_test.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)
# data cleaning
data_test['Age'].fillna(data_test['Age'].median(), inplace=True)
data_test['Cabin'].fillna('XXX', inplace=True)

def reliability(data_pclass, data_sex):
  v_reliability = []
  for i, _pclass in enumerate(data_pclass):
    if _pclass == 1 and data_sex.iloc[i] == 0:
      v_reliability.append(1)
    else:
      v_reliability.append(0)
  return v_reliability

Data_Combination = ['Sex', 'Parch', 'SibSp', 'Cabin', 'Embarked']
label = preprocessing.LabelEncoder()
for i in Data_Combination:
  data = data_train[i].append(data_test[i])
  label.fit(data.values)
  data_train[i] = label.transform(data_train[i])
  data_test[i] = label.transform(data_test[i])

data_train.dropna(inplace=True)
data_train['train'] = reliability(data_train['Pclass'], data_train['Sex'])
X = np.array(data_train.drop('Survived', 1))
y = np.array(data_train['Survived'])
# random forest classifier
clf = ensemble.RandomForestClassifier()
clf.fit(X, y)

data_test.fillna(-99999, inplace=True)
data_test['test'] = reliability(data_test['Pclass'], data_test['Sex'])

testing = np.array(data_test)

# result
data_output = pd.DataFrame()
data_output['PassengerId'] = data_transfer['PassengerId']
data_output['Survived'] = clf.predict(testing).astype(int)
data_output[['PassengerId', 'Survived']].to_csv('D:/Github/titanic_sklearn/resources/titanic_result.csv', index=False)
