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

#* getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. 
# The parameter 'token' is already stemmed. (It means you should not perform stemming inside this function.) Note the 
# differences between getidf("hispan") and getidf("hispanic"). 

def getidf(token):
    n = len(tf_token) # total numbe of documents
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
        for token in tf_token[filename]:
            if cosine_length != 0:
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
        return norm_tf_idf[filename][token]
    else:
        return 0


def make_norm_query_vec(q_vec,q_cos_length):
    q_vec_norm ={}
    for token in q_vec:
        q_vec_norm[token] = q_vec[token]/q_cos_length
    return q_vec_norm
  

def make_posting_list():
    for file in norm_tf_idf:
        for  token in norm_tf_idf[file]:
            if token not in postings_list:
                postings_list[token]= Counter()
            postings_list[token][file] = norm_tf_idf[file][token]
            
def purity_doc(doc, top_ten_doc):
    for token in top_ten_doc:
        if doc not in top_ten_doc[token]:
            return 0                 
    return 1
            
        


#*query(qstring): return a tuple in the form of (filename of the document, score), where the document is the query answer with respect 
# to "qstring" . If no document contains any token in the query, return ("None",0). If we need more than 10 elements from each posting list,
#  return ("fetch more",0)
def query(qstring):
    q_cos_length = 0
    q_vec = {}
    top_ten_doc = {}
    upper_bounds={}
    cosine_similarity = Counter()
    
    qstring.lower()
    make_norm_tf_idf()
    make_posting_list()


    for token in qstring.split():
        token_stem = stemmer.stem(token)
        if token_stem not in postings_list: #If the token 𝑡 doesn't exist in the corpus, ignore it.
            continue
        top_ten_doc[token_stem], bound_wt = zip(*postings_list[token].most_common(10))  #For each token 𝑡 in the query, return the top-10 elements in its corresponding postings list
        upper_bounds[token_stem]=bound_wt[9]

        q_vec[token_stem] = 1+ math.log10(qstring.count(token))
        q_cos_length += q_vec[token_stem]* q_vec[token_stem]
    q_cos_length = math.sqrt(q_cos_length)  
      
    q_vec_norm = make_norm_query_vec(q_vec,q_cos_length)    


    
    for doc in norm_tf_idf:
        sim = 0.0
        for tok in top_ten_doc:
            if doc in top_ten_doc[tok]:
                sim += (q_vec[tok]/q_cos_length) * postings_list[tok][doc]
            else:
                sim = sim+ q_vec[tok]/q_cos_length * upper_bounds[tok]
        cosine_similarity[doc] = sim
    
    best_doc = cosine_similarity.most_common(1)
    
    document, weight = zip(*best_doc)
    doc = document[0]
    
    if best_doc ==0:
        return "None", 0.0000000
    elif purity_doc(document[0],top_ten_doc):
        return doc, weight[0]
  
    else:
        return "Fetch more", 0
              
    
    
   
print("(%s, %.12f)" % query("terror attack"))

'''
query("november december")

print ("idf of reason =%.12f" %getidf("reason"))
print ("idf of hispan =%.12f" %getidf("hispan"))
print ("idf of hispanic =%.12f" %getidf("hispanic"))

print("%.12f" % getweight("2012-10-03.txt","health"))
print("%.12f" % getweight("1960-10-21.txt","reason"))
print("%.12f" % getweight("1976-10-22.txt","agenda"))
print("%.12f" % getweight("2012-10-16.txt","hispan"))
print("%.12f" % getweight("2012-10-16.txt","hispanic"))
'''



    
    
    
