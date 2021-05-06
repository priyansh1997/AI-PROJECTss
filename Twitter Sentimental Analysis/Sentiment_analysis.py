import numpy as np
import pandas as pd
from sklearn.model_selection import  train_test_split

import nltk #It contains text processing libraries for tokenization, parsing, classification, stemming, tagging and semantic reasoning.
from nltk.corpus import stopwords
from nltk.classify import SklearnClassifier

from wordcloud import WordCloud,STOPWORDS
import matplotlib.pyplot  as  plt

df=pd.read_csv("E://AI//NLP//Sentiment.csv")
data=df[['text','sentiment']]


train,test=train_test_split(data,test_size=0.2)
train=train[train.sentiment!="Neutral"]

train_positive=train[train['sentiment']=='Positive']
train_positive=train_positive['text']
train_negetive=train[train['sentiment']=="Negative"]
train_negetive=train_negetive['text']


def wordcloud_draw(data,color='yellow'):
    words=' '.join(data)
    cleaned_words=" ".join([word for word in words.split()
                            if 'http' not in word
                            and not word.startswith('@')
                            and not word.startswith('#')
                            and word!='RT'
                           ])
    wordcloud=WordCloud(stopwords=STOPWORDS, background_color=color,width=2500,height=2000).generate(cleaned_words)
    plt.figure(1,figsize=(13,13))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()
    
print("Positive words")
wordcloud_draw(train_positive,'white')
print("Negative words")
wordcloud_draw(train_negetive)


tweets=[]
stopwords_set=set[stopwords.words("english")]

for idx,row in train.iterrows():
    words_filtered = [e.lower() for e in row.text.split() if len(e) >= 3]
    words_cleaned=[word for word in words_filtered
                   if 'http' not in word
                   and not word.startswith('@')
                   and not word.startswith('#')
                   and word!='RT']
    
    words_without_stopwords=[word for word in words_cleaned
                             if not word in stopwords_set]
    tweets.append((words_without_stopwords, row.sentiment))
    
test_positive=test[test['sentiment']=='Positive']
test_positive=test_positive['text']
test_negative=test[test['sentiment']=='Negative']
test_negative=test_negative['text']

def words_in_tweets(tweets):
    all_words=[]
    for(words,sentiment) in tweets:
        all_words.extend(words)
    return all_words
    
def word_features(wordlist):
    wordlist=nltk.FreqDist(wordlist)
    features=wordlist.keys()
#A frequency distribution records the number of times 
#each outcome of an experi- ment has occured. 
#For example, a frequency distribution could be 
#used to record the frequency of each word type in a document.
#Frequency distributions are encoded by the FreqDist class,
#which is defined by the nltk. probability module.
#The keys() method returns a view object. 
#The view object contains the keys of the dictionary, as a list.
    return features

w_features = word_features(words_in_tweets(tweets))    

def extract_features(document):
    document_words =[document]
    features = {}
    for word in w_features:
        features['contains(%s)' % word] = (word in document_words)
    return features   


training_set=nltk.classify.apply_features(extract_features,tweets)
classifier=nltk.NaiveBayesClassifier.train(training_set)

neg_cnt = 0
pos_cnt = 0
for obj in test_negative: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Negative'): 
        neg_cnt = neg_cnt + 1
for obj in test_positive: 
    res =  classifier.classify(extract_features(obj.split()))
    if(res == 'Positive'): 
        pos_cnt = pos_cnt + 1
        
print('[Negative]: %s/%s '  % (len(test_negative),neg_cnt))        
print('[Positive]: %s/%s '  % (len(test_positive),pos_cnt))    






















































