# Importing Libraries
import numpy as np
import pandas as pd

# Importing Data
data= pd.read_csv('data_edit.csv')
# removing blank rows
data = data.dropna(axis=0, subset=['body'])
# taking sample of data
data2=data['body'][0:1000]
data2=pd.DataFrame(data2)

# NLP- tokenization, stemming, removing Stopwords
import re
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
ps= PorterStemmer()
from nltk.corpus import stopwords

# one mail at a time
def document(email):
    email = re.sub('[^a-zA-Z]',' ',email)  # Removing all punctuations
    email = email.lower()  # LowerCase 
    email= email.split()   # List conversion
    #print(email)  --> Tokenization done on email(punctuations removed)
    email = [ps.stem(word) for word in email if word not in set(stopwords.words('english'))]  # stemming with removal of STOPWORDS
    #print(email)  --> Stemming Done
    return email

# builds a list of all emails
emails = []
for i in range(1000):
    email=data2['body'][i]
    emails.append(document(email))  # all cleaned emails

number_of_emails = len(data2[['body']])

# builds a corpus- Collecting all unique words...
words=[]
for i in range(1000):
    words += emails[i]
words = list(set(words))
number_of_words = len(words)


# construction of the bag of words matrix
bag_of_words  = np.zeros((number_of_emails, number_of_words))
for i in range(number_of_emails):
    for j in range(number_of_words):
        bag_of_words[i, j] = emails[i].count(words[j])

#sentiwordnet
nltk.download('averaged_perceptron_tagger')
nltk.download('sentiwordnet')
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

def get_sentiment(word,tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag not in (wn.NOUN, wn.ADJ, wn.ADV):
        return []
    synsets = wn.synsets(word)
    if not synsets:
        return []
    # Take the first sense, the most common
    synset = synsets[0]
    swn_synset = swn.senti_synset(synset.name())
    return([swn_synset.pos_score(),swn_synset.neg_score(),swn_synset.obj_score()])


ps = PorterStemmer()
words_data = words

import numpy as np
pos_val = nltk.pos_tag(words_data)
tagss=[]
for i in range(14498):
    tagss.append(pos_val[i][1])
lis2=[]
lis3=[]
for x, y in pos_val:
    lis=(get_sentiment(x,y))
    
    if len(lis)==0:
        lis2.append('neutral')
        lis3.append(0)
    else:
        if(lis[0]>lis[1]):
            lis2.append('positive')
            lis3.append(lis[0])
        elif(lis[1]>lis[0]):
            lis2.append('negitive')
            lis3.append(lis[1])
        elif(lis[1]==lis[0]):
            lis2.append('neutral')
            lis3.append(lis[2])
pos=0
for i in lis2:
    if i=='positive':
        pos+=1
print(pos) # Total number of positive words

neg=0
for i in lis2:
    if i=='negitive':
        neg+=1
print(neg) # Total number of negative words

# Example scores for first 10 Emails- (Positive,Negative,Objective)
for x, y in pos_val[0:30]:
    print(get_sentiment(x,y))   # First 10 words with Positive, negative and objectve scores

words_scores=pd.DataFrame(words)
words_scores['Category']=lis2
words_scores['Scores']=lis3
words_scores['Tags']=tagss
print(words_scores.head())  # printing words with positive/negative/neutral

# number of clusters is set to be 3
K = 3

# PLSA
def calculate_pzdw(pwz, pzd):
    number_of_words, K = pwz.shape
    number_of_emails = pzd.shape[1]
    
    pzdw = np.zeros((K, number_of_words, number_of_emails))

    for i in range(number_of_emails):
            pzdw[:, :, i] = (pwz * pzd[:, i]).T

    for i in range(number_of_emails):
            denom = pzdw[:, :, i].sum(axis=0)
            pzdw[:, :, i] /= denom

    return pzdw

def calculate_pwz(pzdw, bag_of_words):
    K, number_of_words, number_of_emails = pzdw.shape

    pwz = np.zeros((number_of_words, K))
 
    for j in range(number_of_words):
        pwz[j] = (bag_of_words[:, j] * pzdw[:, j, :]).sum(axis=1)

    for k in range(K):  
        denom = (bag_of_words * pzdw[k].T).sum()
        pwz[:, k] /= denom

    return pwz

def calculate_pzd(pzdw, bag_of_words):
    K, number_of_words, number_of_emails = pzdw.shape

    pzd = np.zeros((K, number_of_emails))

    for k in range(K):
        pzd[k] = (bag_of_words * pzdw[k].T).sum(axis=1)

    for k in range(K):
        pzd[k] /= bag_of_words.sum(axis=1)

    return pzd

def plsa(bag_of_words, K, number_of_iterations=10, epsilon=0.0001):
    number_of_emails, number_of_words = bag_of_words.shape
    
    # pwz and pzd are randomly initialized
    pwz = np.random.rand(number_of_words, K)
    pzd = np.random.rand(K, number_of_emails)

    # matrices are normalized to attain probabilities 
    normalize = pwz.sum(axis=0)
    pwz /= normalize

    normalize = pzd.sum(axis=0)
    normalize.shape = (1, number_of_emails)
    pzd /= normalize

    for i in range(number_of_iterations):
        last_pwz = np.copy(pwz)

        #E step
        pzdw = calculate_pzdw(pwz, pzd)

        #M step
        pwz = calculate_pwz(pzdw, bag_of_words)
        pzd = calculate_pzd(pzdw, bag_of_words)

        pwz_change = ((last_pwz - pwz) ** 2).sum()

        if pwz_change < epsilon:
            break
        
    return pwz, pzd


pwz, pzd = plsa(bag_of_words, K)

# Sum of Probabilities: Cluster-wise
class0=[]
class1=[]
class2=[]
sum0 = 0
sum1 = 0
sum2 = 0
terms = sorted(enumerate(pwz[:, 0]), key=lambda x: x[1], reverse=True)
for term_id, score in terms[:4000]:
    class0.append("{word:25}".format(word=words[term_id]))
    sum0=sum0+(float)("{score:45}".format(score=str(score)))
print(sum0) # Sum of probabilities of TOPIC 0

terms = sorted(enumerate(pwz[:, 1]), key=lambda x: x[1], reverse=True)
for term_id, score in terms[:4000]:
    class1.append("{word:25}".format(word=words[term_id]))
    sum1=sum1+(float)("{score:45}".format(score=str(score)))
print(sum1) # Sum of probabilities of TOPIC 1

terms = sorted(enumerate(pwz[:, 2]), key=lambda x: x[1], reverse=True)
for term_id, score in terms[:4000]:
    class2.append("{word:25}".format(word=words[term_id]))
    sum2=sum2+(float)("{score:45}".format(score=str(score)))
print(sum2) # Sum of probabilities of TOPIC 2

# List of words in each cluster
for i in range(4000):
    class0[i]=class0[i].strip()
    class1[i]=class1[i].strip()
    class2[i]=class2[i].strip()
        
# Class-wise allotment of particular Email
listt=[]
for i in range(1000):
    c1=0
    c2=0
    c3=0
    c4=0
    email=data2['body'][i]
    email = re.sub('[^a-zA-Z]',' ',email)
    email = email.lower()
    email= email.split()
    email = [ps.stem(word) for word in email if word not in set(stopwords.words('english'))]
    for j in email:
        if j in class0:
            c1+=1
        if j in class1:
            c2+=1
        if j in class2:
            c3+=1
        else:
            c4+=1
    if(c1>c3 and c1>c2 and c1>c4):
        listt.append('0')
    elif(c2>c3 and c2>c1 and c2>c4):
        listt.append('1')
    elif(c3>c1 and c3>c2 and c3>c4):
        listt.append('2')
    else:
        listt.append('3')



# Positive-Negative Classification 
pos_words = open('positive.txt').read().split("\n")
neg_words = open('negative.txt').read().split("\n")
listt2=[]
listt3=[]
for i in range(1000):
    pos_counter=0
    neg_counter=0
    email=data2['body'][i]
    email = re.sub('[^a-zA-Z]',' ',email)
    email = email.lower()
    email= email.split()
    email = [ps.stem(word) for word in email if word not in set(stopwords.words('english'))]
    for j in email:
        if j in pos_words:
            pos_counter += 1
        elif j in neg_words:
            neg_counter += 1
    if(pos_counter>neg_counter):
        listt2.append('Positive')
    elif(pos_counter<neg_counter):
        listt2.append('Negitive')
    else:
        listt2.append('Neutral')
    email=' '.join(email)
    listt3.append(email)

# New data after labelling
data3=pd.DataFrame(listt3)
data3['Class']=listt    # Storing in Dataset, which will be output afterwards
data3['Category']=listt2


def document(email):
    email= re.sub('\n','',email)
    email = re.sub('[^a-zA-Z]',' ',email)  # Removing all punctuations
    email = email.lower()  # LowerCase 
    email= email.split()   # List conversion
    #print(email)  --> Tokenization done on email(punctuations removed)
    email = [ps.stem(word) for word in email if word not in set(stopwords.words('english'))]  # stemming with removal of STOPWORDS
    #print(email)  --> Stemming Done
    email= ' '.join(email)
    return email

# builds a list of all emails
emails = []
for i in range(1000):
    email=data2['body'][i]
    emails.append(document(email))  # all cleaned emails

# Feature Extraction
from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(emails).toarray()
X_train_counts.shape

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# Output Variable
y=data3.iloc[:,2].values

# Label Encoding
from sklearn.preprocessing import LabelEncoder
le_y=LabelEncoder()
y=le_y.fit_transform(y)

# Training and Testing Set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_train_counts,y,test_size=0.1,random_state=0)
 
# XGBoost Algorithm
import xgboost
from xgboost import XGBClassifier
classi= XGBClassifier()
classi.fit(X_train,y_train)
y_pred= classi.predict(X_test)

# Evaluation metrics
from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))





