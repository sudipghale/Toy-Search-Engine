"""
Autho @Sudip Ghale

"""

'''
implementation of a toy "search engine" in Python. 
the program reads a corpus and produce TF-IDF vectors for documents in the corpus. 
Then, given a query string, program returns the query answer--the document with the highest cosine similarity score for the query. 
Instead of computing cosine similarity score for each and every document,  a smarter threshold-bounding algorithm is implemented  
which shares the same basic principle as real search engines like "Google"
'''
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
    doc = doc.lower() # converting all doc to lower case
    tokens = tokenizer.tokenize(doc)
    tokens = [stemmer.stem(word) for word in tokens if word not in stop_words_eng] # a token is stemmed iff token is not in the eng stop word list
    tf_tok = Counter(tokens)
    tf_token[filename] = tf_tok.copy()
    df_token += Counter(set(tokens))

#* getidf(token): return the inverse document frequency of a token. If the token doesn't exist in the corpus, return -1. 
# The parameter 'token' is already stemmed. (It means you should not perform stemming inside this function.) Note the 
# differences between getidf("hispan") and getidf("hispanic"). 

def getidf(token):
    n = len(tf_token) # total numbe of documents
    if df_token[token] == 0: # if token is not not in any documents
        return -1
    else:
        return math.log10(n/df_token[token])  
    
# returns the tf-weight of the token in a file
def get_tf_weight(filename, token):
    return (1+math.log10(tf_token[filename][token]))
#returns the tf-idf weight token in a file
def get_tf_idf_weight (filename, token):
    return (get_tf_weight(filename,token)* getidf(token))
#returns the cosine length of the file
def get_cosine_length(filename):
    cosine_length = 0
    for token in tf_token[filename]:
        cosine_length += get_tf_idf_weight(filename, token) * get_tf_idf_weight(filename, token)    
    return math.sqrt(cosine_length)
# normalzes the tf_idf weights         
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

# For each token 𝑡 that exists in the corpus, construct its postings list---a sorted list in which each element is in the 
# form of (document 𝑑, TF-IDF weight 𝑤). Such an element provides 𝑡's weight 𝑤 in document 𝑑. The elements in the list are sorted by weights in descending order. 
def make_posting_list():
    for file in norm_tf_idf:
        for  token in norm_tf_idf[file]:
            if token not in postings_list:
                postings_list[token]= Counter()
            postings_list[token][file] = norm_tf_idf[file][token]
            
# returns 1 if the document exist in all the top 10 list of all the token, otherwise returns 0
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
        token_stem = stemmer.stem(token) # stem the query token
        if token_stem not in postings_list: #If the token 𝑡 doesn't exist in the corpus, ignore it.
            continue
        top_ten_doc[token_stem], bound_wt = zip(*postings_list[token_stem].most_common(10))  #For each token 𝑡 in the query, return the top-10 elements in its corresponding postings list
        upper_bounds[token_stem]=bound_wt[9] # adding the 10th weight as a upper bound 

        q_vec[token_stem] = 1+ math.log10(qstring.count(token)) # calculating the tf-weight for the query tokens
        q_cos_length += q_vec[token_stem]* q_vec[token_stem] 
    q_cos_length = math.sqrt(q_cos_length)  # calculating the consine lenght for query 
         
    for doc in norm_tf_idf:
        sim = 0.0
        for tok in top_ten_doc:
            if doc in top_ten_doc[tok]:
                sim += (q_vec[tok]/q_cos_length) * postings_list[tok][doc] # calculating the actual cosine similariy scores  
            else:
                sim = sim+ q_vec[tok]/q_cos_length * upper_bounds[tok] # calculating the cosine similarity using the upper bound
        cosine_similarity[doc] = sim
    
    best_doc = cosine_similarity.most_common(1)     # returns the doc with highest cosine similarity
    doc= best_doc[0][0] # extracting the doc string 
    weight = best_doc[0][1] # extracting the doc cosine similarity weight 
        
      
    if weight ==0:
        return "None", 0.0000000  # If no document contains any token in the query, return ("None",0). 
    elif purity_doc(doc,top_ten_doc):
        return doc, weight # if top doc is pure retuns doc and cosine score
  
    else:
        return "Fetch more", 0 #If we need more than 10 elements from each posting list, return ("fetch more",0).
              
# test cases to evaluate the correctness of the program, side commands are expected values
print("%.12f" % getidf("health")) # 0.079181246048
print("%.12f" % getidf("agenda")) #0.363177902413
print("%.12f" % getidf("vector")) #-1.000000000000
print("%.12f" % getidf("reason")) #0.000000000000
print("%.12f" % getidf("hispan")) #0.632023214705
print("%.12f" % getidf("hispanic")) #-1.000000000000
print("%.12f" % getweight("2012-10-03.txt","health")) #0.008528366190
print("%.12f" % getweight("1960-10-21.txt","reason")) #0.000000000000
print("%.12f" % getweight("1976-10-22.txt","agenda")) #0.012683891289
print("%.12f" % getweight("2012-10-16.txt","hispan")) #0.023489163449
print("%.12f" % getweight("2012-10-16.txt","hispanic")) #0.000000000000
print("(%s, %.12f)" % query("health insurance wall street")) #(2012-10-03.txt, 0.033877975254)
print("(%s, %.12f)" % query("particular constitutional amendment")) #(fetch more, 0.000000000000)
print("(%s, %.12f)" % query("terror attack")) #(2004-09-30.txt, 0.026893338131)
print("(%s, %.12f)" % query("vector entropy")) #(None, 0.000000000000)



    
    
    
