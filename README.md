# Toy-Search-Engine
implementing a toy "search engine" in Python using NLTK library

The program reads a corpus and produce TF-IDF vectors for documents in the corpus. 
Then, given a query string, program returns the query answer--the document with the highest cosine similarity score for the query. 
Instead of computing cosine similarity score for each and every document,  a smarter threshold-bounding algorithm is implemented  
which shares the same basic principle as real search engines like "Google"
