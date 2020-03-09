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
norm_tf_idf = {}
postings_list = {} # (doc d, TF-IDF w)


corpusroot = './presidential_debates'
tokenizer = RegexpTokenizer(r'[a-zA-Z]+') 
stemmer = PorterStemmer()
stop_words_eng = stopwords.words("english")

for filename in os.listdir(corpusroot):
    file = open(os.path.join(corpusroot, filename), "r", encoding='UTF-8')
    doc = file.read()
    file.close() 
    doc = doc.lower()
    tokens = tokenizer.tokenize(doc)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_eng] # stop first or stemmin ???
    tf_tok = Counter(tokens)
    tf_token[filename] = tf_tok.copy()
    df_token = df_token + Counter(set(tokens))

#print(tf_token)
#print(df_token)    

#* getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. 
# The parameter 'token' is already stemmed. (It means you should not perform stemming inside this function.) Note the 
# differences between getidf("hispan") and getidf("hispanic"). 

def getidf(token):
    n = len(tf_token) # total numbe of documents
  #  print("N is =", n)
    if df_token[token] == 0: # if token is not not in any documents
        return -1
    else:
        return math.log10(n/df_token[token])  
    

def get_tf_weight(filename, token):
    return (1+math.log10(tf_token[filename][token]))

def get_tf_idf_weight (filename, token):
    return (get_tf_weight(filename,token)* getidf(token))

def get_cosine_length(filename):
    cosine_length = 0
    for token in tf_token[filename]:
        cosine_length += get_tf_idf_weight(filename, token) * get_tf_idf_weight(filename, token)    
    return math.sqrt(cosine_length)
        
def make_norm_tf_idf():
    for filename in tf_token:
        norm_tf_idf[filename] = Counter()
        cosine_length = get_cosine_length(filename)
       # print("cos lengh ===", cosine_length)
        for token in tf_token[filename]:
            if cosine_length != 0:
              #  print("tfidf and cos len of {}  is {}{} ", token, get_tf_idf_weight(filename,token),cosine_length)
                norm_tf_idf[filename][token] = get_tf_idf_weight(filename,token)/cosine_length
            else:
                print("cosine_length is ZERO")
  
#* getweight(filename,token): return the TF-IDF weight of a token in the document named 'filename'. 
# If the token doesn't exist in the document, return 0. The parameter 'token' is already stemmed. 
# (It means you should not perform stemming inside this function.) Note that both getweight("1960-10-21.txt","reason") 
# and getweight("2012-10-16.txt","hispanic") return 0, but for different reasons. 

def getweight(filename, token): #tf-idf weight
    make_norm_tf_idf()
    if any(token in word for word in tf_token[filename]):        
        #print("token found in doc with tf-idf score")
        return norm_tf_idf[filename][token]
    else:
        return 0
        
            
"""
def print_tf_idf(filename, doc_fre):
    for tok in doc_fre:
        print("tf-idf of", tok,"is", getweight(filename, tok));
        print_tf_idf("file1.txt", df_token)


  """
#getweight("file2.txt", "septemb")
#print("the norm_tf_idf is",norm_tf_idf)
 
def make_norm_query_vec(q_vec,q_cos_length):
    q_vec_norm ={}
    for token in q_vec:
        q_vec_norm[token] = q_vec[token]/q_cos_length
    return q_vec_norm
  

def make_posting_list():
    for file in norm_tf_idf:
        postings_list[file] = Counter()
        for  token in norm_tf_idf[file]:
            if token not in postings_list:
                postings_list[file][token]=norm_tf_idf[file][token]

#*query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer with respect 
# to "qstring" . If no document contains any token in the query, return ("None",0). If we need more than 10 elements from each posting list,
#  return ("fetch more",0)
def query(qstring):
    q_cos_length = 0
    q_vec = {}
    
    qstring.lower()
    
    for token in qstring.split():
        token_stem = stemmer.stem(token)
        q_vec[token_stem] = 1+ math.log10(qstring.count(token))
        q_cos_length += q_vec[token_stem]* q_vec[token_stem]
    q_cos_length = math.sqrt(q_cos_length)  
      
    q_vec_norm = make_norm_query_vec(q_vec,q_cos_length)    
    print("q_vec",q_vec_norm)
    #print("tfidf:",norm_tf_idf)
    
    make_posting_list()
    print("posting list= ",postings_list)
    
query("particular constitutional amendment")
   

'''
print ("idf of reason =%.12f" %getidf("reason"))
print ("idf of hispan =%.12f" %getidf("hispan"))
print ("idf of hispanic =%.12f" %getidf("hispanic"))

print("%.12f" % getweight("2012-10-03.txt","health"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
'''



    
    
    
    

