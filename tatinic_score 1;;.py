#!/usr/bin/env python
# coding: utf-8

# In[59]:


import numpy as np
import pandas as pd
import io
import requests
import re
import warnings
import sklearn


# In[60]:


import plotly.express as px

import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls

import seaborn as sns
from sklearn.ensemble import (RandomForestClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, ExtraTreesClassifier)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.svm import SVC


# In[61]:


train_data = pd.read_csv("C:/Users/chlwl/Desktop/train.csv")
df_train = pd.read_csv("C:/Users/chlwl/Desktop/train.csv")
train_data.head()


# In[62]:


test_data = pd.read_csv("C:/Users/chlwl/Desktop/test.csv")
test_data.head()


# In[63]:


# Print train and test columns
test = pd.read_csv("C:/Users/chlwl/Desktop/test.csv")
train = pd.read_csv("C:/Users/chlwl/Desktop/train.csv")
print('Train columns:', train.columns.tolist())
print('Test columns:', test.columns.tolist())


# In[64]:


#This is test file from titanic and gender_submission combined
tested = pd.read_csv("C:/Users/chlwl/Desktop/tested.csv")
tested.head()


# In[65]:


PassengerId = test['PassengerId']
train['Ticket_type'] = train['Ticket'].apply(lambda x: x[0:3])
train['Ticket_type'] = train['Ticket_type'].astype('category')
train['Ticket_type'] = train['Ticket_type'].cat.codes

test['Ticket_type'] = test['Ticket'].apply(lambda x: x[0:3])
test['Ticket_type'] = test['Ticket_type'].astype('category')
test['Ticket_type'] = test['Ticket_type'].cat.codes


# In[66]:


y_train = train['Survived'].ravel()
train = train.drop(['Survived'], axis=1)
x_train = train.values
x_test = test.values 


# In[67]:


full_data = []


# In[68]:


for dataset in full_data:
    dataset['Title'] = dataset['Name'].apply(get_title)
# Group all non-common titles into one single grouping "Rare"
for dataset in full_data:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')


# In[69]:


train.head()


# In[70]:


gender_data = pd.read_csv("C:/Users/chlwl/Desktop/gender_submission.csv")
gender_data.head()


# In[71]:


gender = pd.read_csv("C:/Users/chlwl/Desktop/gender_submission.csv")
print('Gender columns:', gender.columns.tolist())


# In[72]:


women = train_data.loc[train_data.Sex == 'female']["Survived"]
rate_women = sum(women)/len(women)

print("% of women who survived:", rate_women)

##Women and children were the first to board the titanic which means they are more likely to survive than men


# In[73]:


men = train_data.loc[train_data.Sex == 'male']["Survived"]
rate_men = sum(men)/len(men)

print("% of men who survived:", rate_men)


# In[74]:


##The younger you are the more likely to survive

data = [train_data, test_data]
for dataset in data:
    mean = train_data["Age"].mean()
    std = test_data["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    # compute random numbers between the mean, std and is_null
    rand_age = np.random.randint(mean - std, mean + std, size = is_null)
    # fill NaN values in Age column with random values generated
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_data["Age"].astype(int)


# In[75]:


survived = 'survived'
not_survived = 'not survived'
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(16, 8))
women = train_data[train_data['Sex']=='female']
men = train_data[train_data['Sex']=='male']
ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False, color="green")
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False, color="red")
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False, color="green")
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False, color="red")
ax.legend()
_ = ax.set_title('Male');


# In[76]:


df = pd.read_csv('C:/Users/chlwl/Desktop/train.csv')
fig = px.scatter_3d(df, x='PassengerId', y='Sex', z='Age',
                    color='Age')
fig.show()


# In[77]:


df = pd.read_csv('C:/Users/chlwl/Desktop/tested.csv')

for template in ["plotly"]:
    fig = px.scatter(df,
                     x="PassengerId", y="Age", color="Survived",
                     log_x=True, size_max=20,
                     template=template, title="Which Age Survived?")
    fig.show()


# In[78]:


##You have a higher chance of surviving if you have a first class ticket than having a second or third

sns.barplot(x='Pclass', y='Survived', data=train_data);


# In[79]:


plt.rc('xtick', labelsize=14) 
plt.rc('ytick', labelsize=14)

plt.figure()
fig = df_train.groupby('Survived')['Pclass'].plot.hist(histtype= 'bar', alpha = 0.8)
plt.legend(('Died','Survived'), fontsize = 12)
plt.xlabel('Pclass', fontsize = 18)
plt.show()


# In[80]:


## Women will survive more if they embarked from port ‘Southampton’ or 
##‘ Queenstown’. While men will survive more from the port 'Cherbourg'. 
## Passengers from port ‘Southampton’ have a low survival rate of 34%, while those from the port ‘Cherbourg’ have a survival rate of 55%. Over 72% of the passengers embarked from the port 'Southampton’, 
## 18% from the port ‘Cherbourg’ and the rest from the port ‘Queenstown’

embarked_mode = train_data['Embarked'].mode()
data = [train_data, test_data]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)


# In[81]:


FacetGrid = sns.FacetGrid(train_data, row='Embarked', size=4.5, aspect=1.6)
FacetGrid.map(sns.pointplot, 'Pclass', 'Survived', 'Sex', order=None, hue_order=None )
FacetGrid.add_legend();


# In[82]:


sns.set(style="darkgrid")
sns.countplot( x='Survived', data=train_data, hue="Embarked", palette="Set1");


# In[83]:


## You are more likly to survive if you are travels with 1 to 3 people and if you have 0 or more than three you have a less chance.
data = [train_data, test_data]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'
    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'
axes = sns.factorplot('relatives','Survived', 
                      data=train_data, aspect = 2.5, );


# In[84]:


## Shows the number of females and males who has number of siblings or spouse that is Parch.

df = pd.read_csv("C:/Users/chlwl/Desktop/train.csv")
fig = px.histogram(df, x="SibSp", y="Parch", color="Sex", marginal="rug",
                   hover_data=df.columns)
fig.show()


# In[85]:


##train_numerical_features = list(train_data.select_dtypes(include=['int64', 'float64', 'int32']).columns)
##ss_scaler = StandardScaler()
##train_data_ss = pd.DataFrame(data = train_data)
##train_data_ss[train_numerical_features] = ss_scaler.fit_transform(train_data_ss[train_numerical_features])


# In[86]:


param_test1 = {
    'n_estimators': [100,200,500,750,1000],
    'max_depth': [3,5,7,9],
    'min_child_weight': [1,3,5],
    'gamma':[i/10.0 for i in range(0,5)],
    'subsample':[i/10.0 for i in range(6,10)],
    'colsample_bytree':[i/10.0 for i in range(6,10)],
    'reg_alpha':[0, 0.001, 0.005, 0.01, 0.05, 0.1, 1],
    'learning_rate': [0.01, 0.02, 0.05, 0.1]
}


# In[87]:


train_data.head(10)


# In[88]:


## This shows an estimate of the output, so not exact. 1 Shows if the person had survived, while 0 shows that person died.
for template in ["plotly_dark"]:
    fig = px.scatter(df,
                     x="PassengerId", y="Survived", color="Survived",
                     log_x=True, size_max=20,
                     template=template, title="Survived or Died?")
    fig.show()


# In[89]:


from sklearn.ensemble import RandomForestClassifier

#data["Age"] = data["Age"].astype(int)
#if ["Age"]

y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
X = pd.get_dummies(train_data[features])
X_test = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=3, random_state=2)
model.fit(X, y)
predictions = model.predict(X_test)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('submission.csv', index=False)


# In[90]:


##In conclusion

#There are many things for a greater chance to survive. 
#Being a female or a child will increase you chances. 
#If you have a higher class ticket you have the more chance of surviving than a third class ticket. 
#As for a man, you are more likely to survive if embark in Cherbourg compare to Southampton or Queenstown. 
#If you also travel with 1 or 3 people than 0 or more than 3 your survival chances are greater. 
#The younger you are will also make your survival chance. 
#So it comes down to many things to surivive on the titanic.##


# In[ ]:




