import numpy as np
import pandas as pd
import re
import os
import codecs
import nltk
from sklearn import feature_extraction
import mpld3
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

i=0
find_d="find_d"
find_t="find_t"
file = open('test.txt', 'r')

for line in file:
    if i==0:
        find_t=line
        i+=1
    else:
        find_d=line

print find_t
print "......"
print find_d

''''nltk.download("stopwords")
nltk.download('punkt')'''
stemmer = SnowballStemmer('english')
stopWords = set(stopwords.words('english'))

# here I define a tokenizer and stemmer which returns the set of stems in the text that it is passed

def tokenize_and_stem(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def tokenize_only(text):
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    tokens = [word.lower() for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    return filtered_tokens


    
def readFile2():
    with open('hello2.txt') as f:
        titles = f.read().decode('utf-8').strip().splitlines()
        titles.append(find_t)
    return titles


#tfidf_transformer=getAlgorithms();


with open('hello.txt') as f:
    lines = f.read().decode('utf-8').strip().splitlines()
    lines.append(find_d)
totalvocab_stemmed = []
totalvocab_tokenized = []
for i in lines:
    allwords_stemmed = tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
    totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
    
    allwords_tokenized = tokenize_only(i)
    totalvocab_tokenized.extend(allwords_tokenized)

vocab_frame = pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
print 'there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame'

print vocab_frame.head()

print
print 
print


tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                 min_df=0.05, stop_words='english',
                                 use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1,3))
tfidf_matrix = tfidf_vectorizer.fit_transform(lines)
'''print(tfidf_matrix.shape)'''
terms = tfidf_vectorizer.get_feature_names()
print terms




dist = 1 - cosine_similarity(tfidf_matrix)
print
print



num_clusters = 20
km = KMeans(n_clusters=num_clusters)
km.fit(tfidf_matrix)
clusters = km.labels_.tolist()



joblib.dump(km,  'doc_cluster.pkl')

km = joblib.load('doc_cluster.pkl')
clusters = km.labels_.tolist()

films = {  'title':readFile2(),'description': lines, 'cluster': clusters }

frame = pd.DataFrame(films, index = [clusters] , columns = ['title','description', 'cluster'])
frame['cluster'].value_counts()


print("Top terms per cluster:")
print
#sort cluster centers by proximity to centroid
order_centroids = km.cluster_centers_.argsort()[:, ::-1] 

for i in range(num_clusters):
    print "Cluster "+ str(i)+ " words:" 
    for ind in order_centroids[i, :3]: #replace 6 with n words per cluster
        print vocab_frame.ix[terms[ind].split(' ')].values.tolist()[0][0].encode('utf-8', 'ignore')
    print #add whitespace
    print #add whitespace
    print "Cluster " +str(i)+" titles:"  
    for desc in frame.ix[i]['title'].values.tolist():
        print desc