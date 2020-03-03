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

tf_token = {}
df_token = Counter()
stemmed_words =[]

corpusroot = './test'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+') 
stemmer = PorterStemmer()
stop_words_eng = stopwords.words("english")

for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close() 
    doc = doc.lower()
    tokens = tokenizer.tokenize(doc)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_eng]
    tf_tok = Counter(tokens)
    tf_token[filename] = tf_tok.copy()
    df_token = df_token + Counter(set(tokens))

    
    