"""
Sudip Ghale
Data Mining: Lab01
"""
import os
import math
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

tf_token = []
df_token = []
corpusroot = './presidential_debates'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+') 
stemmer = PorterStemmer()


for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close() 
    doc = doc.lower()



tokens = tokenizer.tokenize(doc)
filtered_words = [word for word in tokens if word not in stopwords.words('english')]

print(len(filtered_words))

stemmed_words =[]
for word in filtered_words:
    stemmed_words.append(stemmer.stem(word))

print(len(stemmed_words))

