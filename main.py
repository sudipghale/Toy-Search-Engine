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

print(tf_token)
print(df_token)    

#* getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. 
# The parameter 'token' is already stemmed. (It means you should not perform stemming inside this function.) Note the 
# differences between getidf("hispan") and getidf("hispanic"). 

def getidf(token):
    n = len(tf_token)
    print("N is =", n)
    if df_token[token] == 0:
        return -1
    else:
        return math.log10(n/df_token[token])  
    
#* getweight(filename,token): return the TF-IDF weight of a token in the document named 'filename'. 
# If the token doesn't exist in the document, return 0. The parameter 'token' is already stemmed. 
# (It means you should not perform stemming inside this function.) Note that both getweight("1960-10-21.txt","reason") 
# and getweight("2012-10-16.txt","hispanic") return 0, but for different reasons. 

def getweight(filename, token): #tf-idf weight
    idf_token = getidf(token)
    print("idf of token",idf_token)
    if any(token in word for word in tf_token[filename]):
        tf_idf_token =(1+math.log10(tf_token[filename][token]))* idf_token
        print("token found in doc with tf-idf score", tf_idf_token)
        return tf_idf_token
    else:
        return 0
        
        
#*query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer with respect 
# to "qstring" . If no document contains any token in the query, return ("None",0). If we need more than 10 elements from each posting list,
#  return ("fetch more",0).
        
    
    

tf_idf = getweight("file1.txt", "octob")
print("tf-idf", tf_idf);