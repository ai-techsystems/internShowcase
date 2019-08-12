
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

dataset=pd.read_csv('train.csv')
testdata=pd.read_csv('test.csv')

#below are the graphs and plots used for data preprocessing
'''sns.countplot(x='Survived',data=dataset);

sns.catplot(x='Sex', col='Survived', kind='count', data=dataset)

#sns.catplot(x='Pclass', col='Embarked', kind='count', data=dataset)
sns.catplot(x='Embarked', col='Survived', kind='count', data=dataset)
sns.catplot(x='Sex', col='Embarked', kind='count', data=dataset)


sns.catplot(x='Age', col='Survived', kind='count', data=dataset)

sns.relplot(x="Age", hue="Survived", kind="line", data=dataset)

sns.FacetGrid(row="Age",hue="Survived",data=dataset)

facet = sns.FacetGrid(data = dataset,hue="Survived",legend_out=True,size = 4.5,height=6,aspect=2)
facet = facet.map(sns.kdeplot, "Age")
facet.add_legend();

facet = sns.FacetGrid(data = dataset,hue="Survived",col="Sex",legend_out=True,size = 4.5,height=6,aspect=2)
facet = facet.map(sns.kdeplot, "Age")
facet.add_legend();'''


all_df =pd.concat([dataset,testdata],sort=False)


all_df.isnull().sum()
all_df['Embarked'] = all_df['Embarked'].fillna('C')
all_df['Age']      = all_df['Age'].fillna(all_df['Age'].median())
all_df['Fare']     = all_df['Fare'].fillna(all_df['Fare'].median())
all_df['Cabin']    = all_df['Cabin'].fillna('Z')
all_df['Cabin']    = all_df['Cabin'].apply(lambda x: str(x)[0])

all_df['familysize']=all_df['SibSp']+all_df['Parch']
sns.catplot(x='familysize', col='Survived', kind='count', data=all_df)

sns.barplot(x='familysize',y='Survived',data=all_df)


all_df.loc[(all_df.familysize==0),'FamilyType'] = 2
all_df.loc[((all_df.familysize>=1)&(all_df.familysize<=3)),'FamilyType'] = 3
all_df.loc[(all_df.familysize>=4),'FamilyType'] = 1

'''facet = sns.FacetGrid(data = dataset,hue="Survived",legend_out=True,size = 4.5,height=6,aspect=2)
facet = facet.map(sns.kdeplot, "Fare")
facet.add_legend();


facet = sns.FacetGrid(data = dataset,col='Pclass',hue="Survived",legend_out=True,size = 4.5,height=6,aspect=2)
facet = facet.map(sns.kdeplot, "Fare")
facet.add_legend();

sns.barplot(x='Embarked',y='Survived',data=all_df)
'''
sns.barplot(x='Pclass',y='Survived',data=all_df)
#encoding embarked 
all_df['Embarked']=all_df['Embarked'].map({'S':1,'C':3,'Q':2})

all_df['Cabin'].unique()
#sns.barplot(x='Cabin',y='Survived',data=all_df)
all_df['Cabin']=all_df['Cabin'].replace(['E','B','D'],3)
all_df['Cabin']=all_df['Cabin'].replace(['C','F'],2)
all_df['Cabin']=all_df['Cabin'].replace(['A','G','T'],1)
all_df['Cabin']=all_df['Cabin'].replace(['Z'],0)

all_df['Title']=all_df['Name'].apply(lambda x: str(x).split(',')[1].split('.')[0])
all_df['Title'].unique()

#sns.barplot(y='Title',x='Survived',data=all_df)
all_df['Title']=all_df['Title'].astype('str')
all_df['Title']=all_df['Title'].replace(['Mme','Ms','Lady','Sir','Mlle','the Countess'],3,regex=True)
all_df['Title']=all_df['Title'].replace(['Mrs','Miss'],2,regex=True)
all_df['Title']=all_df['Title'].replace(['Mr','Master','Dr','Major','Col'],1,regex=True)
all_df['Title']=all_df['Title'].replace(['Don','Rev','Capt','Jonkheer','Dona'],0,regex=True)


all_df.loc[(all_df.Age<=16),'Agegroup']=1
all_df.loc[(all_df.Age>16)&(all_df.Age<=40),'Agegroup']=2
all_df.loc[(all_df.Age>40)&(all_df.Age<=60),'Agegroup']=3
all_df.loc[(all_df.Age>60),'Agegroup']=4

#sns.barplot(x='Agegroup',y='Survived',data=all_df)

all_df['Agecode']=all_df['Agegroup'].map({1:3,2:1,3:2,4:0})

#sns.barplot(y='Ticket',x='Survived',data=all_df)

len(all_df['Ticket'].unique())

all_df['Sex']=all_df['Sex'].map({'male':0,'female':1})

all_df['Cabin']=all_df['Cabin'].astype(int)

finaldataset=all_df.copy()
finaldataset=finaldataset.drop(['PassengerId','Name','Age','SibSp','Parch','Fare','Ticket','familysize','Agegroup'],axis=1)

X = finaldataset.iloc[:,1:].values
y = finaldataset.iloc[:, 0].values

X_train=finaldataset.iloc[:len(dataset),1:].values
X_test=finaldataset.iloc[len(dataset):,1:].values
y_train=finaldataset.iloc[:len(dataset),0].values


X_tr=pd.DataFrame(X_train)
X_tr.to_csv("X_train.csv")
X_te=pd.DataFrame(X_test)
X_te.to_csv("X_test.csv")
y_tr=pd.DataFrame(y_train)
y_tr.to_csv("y_train.csv")




