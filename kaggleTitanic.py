
# coding: utf-8

# In[273]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sb
import re
from sklearn import ensemble
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.keras.layers
train=pd.read_csv('input.csv')
test=pd.read_csv('test.csv')

embark_codes = {'S':1,'C':2,'Q':3}

train['E_Codes']=0

train['Embarked'] = train['Embarked'].fillna('C')
train['E_Codes'] = train['Embarked'].map(embark_codes)
# train['E_Codes'] = train['E_Codes'].fillna(0)

test['E_Codes']=0

test['Embarked'] = test['Embarked'].fillna('C')
test['E_Codes'] = test['Embarked'].map(embark_codes)
#     test['E_Codes'] = test['E_Codes'].fillna(0)
    
titles = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Other': 5}


# extract titles
train['Title'] = train.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# replace titles with a more common title or as Other
train['Title'] = train['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                           'Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss',
                            'Miss','Miss','Mr','Mr','Mrs','Mrs','Other',
                            'Other','Other','Mr','Mr','Mr'])


train['Title'] = train['Title'].map(titles)
train['Title'] = train['Title'].fillna(5)

    

# extract titles
test['Title'] = test.Name.str.extract(' ([A-Za-z]+)\.', expand=False)
# replace titles with a more common title or as Other
test['Title'] = test['Title'].replace(['Mlle','Mme','Ms','Dr','Major','Lady','Countess',
                           'Jonkheer','Col','Rev','Capt','Sir','Don'],['Miss',
                            'Miss','Miss','Mr','Mr','Mrs','Mrs','Other',
                            'Other','Other','Mr','Mr','Mr'])

#print train['Title']
test['Title'] = test['Title'].map(titles)
test['Title'] = test['Title'].fillna(5)
#print test['Title']
    
#print train.groupby('Title')['Age'].mean()

"""
def what_is_age():
    for l in train:
        if train['Title']==1:
            return 33
        elif train['train']==2:
            return 22
        elif train['Title']==3:
            return 36
        elif train['Title']==4:
            return 5
        else:
            return 46
            
"""            
            
train.loc[(train.Age.isnull()) & (train.Title==1),'Age']=33
train.loc[(train.Age.isnull()) & (train.Title==3),'Age']=36
train.loc[(train.Age.isnull()) & (train.Title==4),'Age']=5
train.loc[(train.Age.isnull()) & (train.Title==2),'Age']=22
train.loc[(train.Age.isnull()) & (train.Title==5),'Age']=46
train['Age'] = train['Age'].fillna(0)  
 
#print train['Age']

test.loc[(test.Age.isnull()) & (test.Title==1),'Age']=33
test.loc[(test.Age.isnull()) & (test.Title==3),'Age']=36
test.loc[(test.Age.isnull()) & (test.Title==4),'Age']=5
test.loc[(test.Age.isnull()) & (test.Title==2),'Age']=22
test.loc[(test.Age.isnull()) & (test.Title==5),'Age']=46
test['Age'] = test['Age'].fillna(0) 

train['age_group']=0
train.loc[(train.Age<=18),'age_group']=0
train.loc[(train.Age>18)&(train.Age<=35),'age_group']=1
train.loc[(train.Age>35)&(train.Age<=60),'age_group']=2
train.loc[(train.Age>60),'age_group']=3

test['age_group']=0
test.loc[(test.Age<=18),'age_group']=0
test.loc[(test.Age>18)&(test.Age<=35),'age_group']=1
test.loc[(test.Age>35)&(test.Age<=60),'age_group']=2
test.loc[(test.Age>60),'age_group']=3

train['deck']=0


train['deck']=train['Cabin'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()

train['deck'] = train['deck'].fillna(0)
    
    #print train['deck']
test['deck']=0

test['deck']=test['Cabin'].str.extract('([a-zA-Z ]+)', expand=False).str.strip()

test['deck'] = test['deck'].fillna(0)
    
    #print test['deck']



train['room']=0
# for m in train:
train['room']=train['Cabin'].str.extract('(\d+)', expand=False).str.strip()
#     print (train['room'])

    
train['family_size']=0
for l in train:
    train['family_size']=train['Parch']+train['SibSp']+1
    #print train['family_size']
    
    
    
# for z in test:
test['room']=test['Cabin'].str.extract('(\d+)', expand=False).str.strip()
#     print (test['room'])
    
test['family_size']=0
# for f in test:
test['family_size']=test['Parch']+test['SibSp']+1
    #print test['family_size']
    
 


train['Fare_Group']=0
train.loc[(train.Fare<=35),'Fare_Group']=1
train.loc[(train.Fare>35) & (train.Fare<=70),'Fare_Group']=2
train.loc[(train.Fare>70) & (train.Fare<=105),'Fare_Group']=3
train.loc[(train.Fare>105) & (train.Fare<=140),'Fare_Group']=4
train.loc[(train.Fare>140) & (train.Fare<=175),'Fare_Group']=5
train.loc[(train.Fare>175) & (train.Fare<=210),'Fare_Group']=6
train.loc[(train.Fare>210) & (train.Fare<=245),'Fare_Group']=76
train.loc[(train.Fare>245) & (train.Fare<=280),'Fare_Group']=8
train.loc[(train.Fare>280) & (train.Fare<=315),'Fare_Group']=9
train.loc[(train.Fare>315) & (train.Fare<=350),'Fare_Group']=10
train.loc[(train.Fare>350) & (train.Fare<=385),'Fare_Group']=11
train.loc[(train.Fare>385) & (train.Fare<=420),'Fare_Group']=12
train.loc[(train.Fare>420) & (train.Fare<=455),'Fare_Group']=13
train.loc[(train.Fare>455) & (train.Fare<=490),'Fare_Group']=14
train.loc[(train.Fare>490) & (train.Fare<=525),'Fare_Group']=15

train['Fare_Group'] = train['Fare_Group'].fillna(0)  
#print train['Fare_Group']

test['Fare_Group']=0
test.loc[(test.Fare<=35),'Fare_Group']=1
test.loc[(test.Fare>35) & (test.Fare<=70),'Fare_Group']=2
test.loc[(test.Fare>70) & (test.Fare<=105),'Fare_Group']=3
test.loc[(test.Fare>105) & (test.Fare<=140),'Fare_Group']=4
test.loc[(test.Fare>140) & (test.Fare<=175),'Fare_Group']=5
test.loc[(test.Fare>175) & (test.Fare<=210),'Fare_Group']=6
test.loc[(test.Fare>210) & (test.Fare<=245),'Fare_Group']=7
test.loc[(test.Fare>245) & (test.Fare<=280),'Fare_Group']=8
test.loc[(test.Fare>280) & (test.Fare<=315),'Fare_Group']=9
test.loc[(test.Fare>315) & (test.Fare<=350),'Fare_Group']=10
test.loc[(test.Fare>350) & (test.Fare<=385),'Fare_Group']=11
test.loc[(test.Fare>385) & (test.Fare<=420),'Fare_Group']=12
test.loc[(test.Fare>420) & (test.Fare<=455),'Fare_Group']=13
test.loc[(test.Fare>455) & (test.Fare<=490),'Fare_Group']=14
test.loc[(test.Fare>490) & (test.Fare<=525),'Fare_Group']=15


train.loc[(train['deck']==0) & (train.Fare_Group==1) & (train.Pclass==1),'deck']='C'
train.loc[(train['deck']==0) & (train.Fare_Group==1) & (train.Pclass==2),'deck']='F'
train.loc[(train['deck']==0) & (train.Fare_Group==1) & (train.Pclass==3),'deck']='F'

train.loc[(train['deck']==0) & (train.Fare_Group==2) & (train.Pclass==1),'deck']='E'
train.loc[(train['deck']==0) & (train.Fare_Group==2) & (train.Pclass==2),'deck']='F'
train.loc[(train['deck']==0) & (train.Fare_Group==2) & (train.Pclass==3),'deck']='E'


train.loc[(train['deck']==0) & (train.Fare_Group==3) & (train.Pclass==1),'deck']='B'
train.loc[(train['deck']==0) & (train.Fare_Group==3) & (train.Pclass==2),'deck']='B'

train.loc[(train['deck']==0) & (train.Fare_Group==4) & (train.Pclass==1),'deck']='C'

train.loc[(train['deck']==0) & (train.Fare_Group==5) & (train.Pclass==1),'deck']='C'

train.loc[(train['deck']==0) & (train.Fare_Group==6) & (train.Pclass==1),'deck']='B'

train.loc[(train['deck']==0) & (train.Fare_Group==15) & (train.Pclass==1),'room']='B'
train.loc[(train['deck']==0),'deck']='A'


test.loc[(test['deck']==0) & (test.Fare_Group==1) & (test.Pclass==1),'deck']='C'
test.loc[(test['deck']==0) & (test.Fare_Group==1) & (test.Pclass==2),'deck']='F'
test.loc[(test['deck']==0) & (test.Fare_Group==1) & (test.Pclass==3),'deck']='F'

test.loc[(test['deck']==0) & (test.Fare_Group==2) & (test.Pclass==1),'deck']='E'
test.loc[(test['deck']==0) & (test.Fare_Group==2) & (test.Pclass==2),'deck']='F'
test.loc[(test['deck']==0) & (test.Fare_Group==2) & (test.Pclass==3),'deck']='E'


test.loc[(test['deck']==0) & (test.Fare_Group==3) & (test.Pclass==1),'deck']='A'
test.loc[(test['deck']==0) & (test.Fare_Group==3) & (test.Pclass==2),'deck']='A'

test.loc[(test['deck']==0) & (test.Fare_Group==4) & (test.Pclass==1),'deck']='C'

test.loc[(test['deck']==0) & (test.Fare_Group==5) & (test.Pclass==1),'deck']='C'

test.loc[(test['deck']==0) & (test.Fare_Group==6) & (test.Pclass==1),'deck']='B'

test.loc[(test['deck']==0) & (test.Fare_Group==15) & (test.Pclass==1),'room']='B'
test.loc[(test['deck']==0),'deck']='A'

# print (train["deck"].isnull().sum())
train['Age'] = train['Age'].fillna(0)  
    
    
total = train.isnull().sum().sort_values(ascending=False)
#print total

corr_matrix = train.corr() #here we find he correlation between variables
#print corr_matrix['Survived'].sort_values(ascending=False) #relating to lastsoldprice we correlate all the attributes to it

#print train['deck']
#print train['room']
train['alone']=0
train.loc[(train['SibSp']==0)&(train['Parch']==0),'alone']=0
train.loc[(train['SibSp']>0)|(train['Parch']>0),'alone']=1

# print (train['alone'])

test['alone']=0
test.loc[(test['SibSp']==0)&(test['Parch']==0),'alone']=0
test.loc[(test['SibSp']>0)|(test['Parch']>0),'alone']=1
# print (test['alone'])

# n = pd.get_dummies(train.deck)
# n1= pd.get_dummies(test.deck)
# #print Deck
# train = pd.concat([train, n], axis=1)
# test = pd.concat([test, n1], axis=1)
# print (train)
    
#print train['Deck']
#print train.deck.unique()

train = train.drop(['Name','room','Cabin','age_group', 'PassengerId','Ticket','Embarked','Fare_Group','deck','Parch','SibSp','alone'], axis=1)
test = test.drop(['Name','room','Cabin','Ticket', 'age_group', 'PassengerId','Embarked','Fare_Group','deck','SibSp','Parch','alone'], axis=1)

"""    
for z in test:
    test['Ticket']=test['Ticket'].str.extract('(\d+)', expand=False).str.strip()
    print test['Ticket']
    
print test["Ticket"].isnull().sum()
#test['Ticket'] = test['Ticket'].astype(int)
test['Ticket'] = test['Ticket'].astype(int)
for z in train:
    train['Ticket']=train['Ticket'].str.extract('(\d+)', expand=False).str.strip()
    #print train['Ticket']
    train['Ticket'] = train['Ticket'].fillna(0) 
print train["Ticket"].isnull().sum()
train['Ticket'] = train['Ticket'].astype(int)

#train['Ticket'] = train['Ticket'].astype(int)
"""
#print train.isnull().sum().sort_values(ascending=False)


train.loc[(train['Sex']=='male') ,'Sex']=0
train.loc[(train['Sex']=='female') ,'Sex']=1

test.loc[(test['Sex']=='male') ,'Sex']=0
test.loc[(test['Sex']=='female') ,'Sex']=1
train['Sex'] = train['Sex'].astype(int)
corr_matrix = train.corr() #here we find he correlation between variables
print (corr_matrix['Survived'].sort_values(ascending=False)) #relating to lastsoldprice we correlate all the attributes to it


# In[329]:


X = train.drop("Survived", axis=1)
Y = train["Survived"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.05, random_state=42)

#print X_train.columns.values
#print X_test.columns.values
print(X_train.info())
print(X_test.info())

X_train=np.nan_to_num(X_train)
Y_train=np.nan_to_num(Y_train)
X_test=np.nan_to_num(X_test)

train['Age'] = train['Age'].astype(int)
train['Fare'] = train['Fare'].astype(int)

X_Validate = np.nan_to_num(test)
# print(X_Validate.info())


#print np.isnan(X_test.values.any())



#Y_prediction = random_forest.predict(X_test)


# In[330]:


random_forest = RandomForestClassifier(n_estimators=200)
random_forest.fit(X_train, Y_train)




acc_random_forest = round(random_forest.score(X_test, Y_test) * 100, 2)
print(round(acc_random_forest,2,), "%")


# In[336]:


Y_prediction = random_forest.predict(X_Validate)

line="PassengerId,Survived"+"\n"
for i in range(0,len(X_Validate)):
    line+=str(i+892)+","+str(Y_prediction[i]) + "\n"
    
with open("TitanicSubmission2.csv",'w') as f:
    f.write(line)


# In[284]:


from keras import backend as K
K.tensorflow_backend._get_available_gpus()


# In[319]:


model = tf.keras.Sequential()
# Adds a densely-connected layer with 64 units to the model:
model.add(tf.keras.layers.Dense(7, activation='relu',input_dim=7))
model.add(tf.keras.layers.Dropout(0))

model.add(tf.keras.layers.Dense(4, activation='relu'))
model.add(tf.keras.layers.Dropout(0))
model.add(tf.keras.layers.Dense(2, activation='relu'))
model.add(tf.keras.layers.Dropout(0)) 
# Add a softmax layer with 10 output units:
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))


# In[320]:


model.compile(optimizer=tf.train.AdamOptimizer(), 
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[321]:


model.summary()


# In[323]:


model.fit(X_train, Y_train, epochs=3000)


# In[310]:


test_loss, test_acc = model.evaluate(X_test, Y_test)

print('Test accuracy:', test_acc)


# In[297]:


Y_test_pred=model.predict_classes(X_Validate)


# In[298]:


line="PassengerId,Survived"+"\n"
for i in range(0,len(X_Validate)):
    line+=str(i+892)+","+str(Y_test_pred[i][1-1]) + "\n"
   
    
with open("TitanicSubmission2.csv",'w') as f:
    f.write(line)

