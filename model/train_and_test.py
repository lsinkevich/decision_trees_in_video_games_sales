import pandas as pd
import numpy as np
import seaborn as sns
from sklearn import preprocessing, ensemble

sns.set_style('whitegrid')

data_train = pd.read_csv('D:/Github/titanic_sklearn/resources/titanic_train.csv')
#selection of necessary information
data_train = data_train.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)
# fill in the gaps
data_train['Embarked'] = data_train['Embarked'].fillna('S')
# data cleaning
data_train['Cabin'].fillna('cabin', inplace=True)

data_test = pd.read_csv('D:/Github/titanic_sklearn/resources/titanic_test.csv')
#selection of necessary information
data_transfer = data_test.drop(['Name', 'Ticket', 'Fare'], axis=1)
data_test = data_test.drop(['PassengerId', 'Name', 'Ticket', 'Fare'], axis=1)
# data cleaning
data_test['Cabin'].fillna('cabin', inplace=True)

Data_Combination = ['Sex', 'Parch', 'SibSp', 'Cabin', 'Embarked']
label = preprocessing.LabelEncoder()
for i in Data_Combination:
  data = data_train[i].append(data_test[i])
  label.fit(data.values)
  data_train[i] = label.transform(data_train[i])
  data_test[i] = label.transform(data_test[i])

def value_pclass(data_pclass):
  v_pclass = []
  for _pclass in enumerate(data_pclass):
    if _pclass == 1:
      v_pclass.append(1)
    else:
      v_pclass.append(0)
  return v_pclass

data_train.dropna(inplace=True)
data_train['1'] = value_pclass(data_train['Pclass'])
X = np.array(data_train.drop('Survived', 1))
y = np.array(data_train['Survived'])
# random forest classifier
clf = ensemble.RandomForestClassifier()
clf.fit(X, y)

data_test.fillna(-99999, inplace=True)
data_test['2'] = value_pclass(data_test['Pclass'])
test_data = np.array(data_test)

# result
data_result = pd.DataFrame()
data_result['PassengerId'] = data_transfer['PassengerId']
data_result['Survived'] = clf.predict(test_data).astype(int)
data_result[['PassengerId', 'Survived']].to_csv('D:/Github/titanic_sklearn/resources/titanic_result.csv', index=False)
