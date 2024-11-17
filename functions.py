# importing necessary libraries
import nltk
nltk.download('punkt_tab')  #Used for tokenization
nltk.download('wordnet')  #Provides the lexical database needed for lemmatization.
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import string

# Download necessary NLTK data files
nltk.download('stopwords')
nltk.download('punkt')

stop_words = set(stopwords.words('english')) 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()





def preprocessing(string):
    '''
    Preprocesses the input text by tokenizing it, removing stop words, and applying stemming.

    '''

    #ensure text is a string (in case of NaN or other data types)
    if not isinstance(string, str):
        return ""
    
    #to lower case
    string = string.lower()
    # Remove extra whitespace
    string = re.sub(r'\s+', ' ', string).strip()
    #handling compound words with '-' to treat them as separate tokens
    string = re.sub(r'-', ' ', string) 

    #tokenize
    tokens = word_tokenize(string)
    #remove punctuation: keep only aplhanumeric tokens
    tokens = [token for token in tokens if token.isalnum()]
    #remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    #stemming to reduce words to their root in order to reduce our vocabulary size, improve search accuracy and speed up search process
    tokens = [stemmer.stem(token) for token in tokens]

    return tokens




import csv
from collections import defaultdict

def create_vocabulary(descriptions):

    '''
    Builds a vocabulary from the descriptions in the input data.

    '''

    #initialize vocabulary
    vocabulary = {}
    #initialize term_id values
    term_id = 1
    for description in descriptions:
        for word in set(description):  # Use set to avoid duplicates in each description
            if word not in vocabulary and ( word.isdigit() or len(word) > 1):  # Check word length
                vocabulary[word] = term_id
                term_id += 1

    # writing CSV file
    with open('vocabulary.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['term', 'term_id'])
        for term, term_id in vocabulary.items():
            writer.writerow([term, term_id])
    
    return vocabulary



import json

def create_inverted_index(descriptions, vocabulary):
    inverted_index = defaultdict(list)
    '''
    creates the inverted index json file
    
    '''
    
    # Iterate over descriptions (documents) were doc_id is the row index (uniques)
    for doc_id, description in enumerate(descriptions, start=0):

        unique_words = set(description)  # Remove duplicates

        for word in unique_words:
            if word in vocabulary:
                term_id = vocabulary[word]   #map the term to the corresponding document ID
                if doc_id not in inverted_index[term_id]: # to avoid duplicated for the same term in the same document
                    inverted_index[term_id].append(doc_id)
    
    # Save the inverted index
    with open('inverted_index.json', 'w') as file:
        json.dump(inverted_index, file, indent=4)

    return inverted_index


def highlight_matches(result_df, query_words):
    # Go through each row in the result DataFrame
    for idx, row in result_df.iterrows():
        description = row['description']
        matched_words = [word for word in query_words if word in description]
        
        # Print the restaurant name and description with matched words highlighted
        print(f"Restaurant: {row['restaurantName']}")
        print(f"Address: {row['address']}")
        
        highlighted_description = description
        for word in matched_words:
            highlighted_description = highlighted_description.replace(word, f"**{word}**")
        
        print(f"Description: {highlighted_description}")
        print(f"Website: {row['website']}\n")




import math
from collections import Counter

def tf_idf_inverted_index(data, vocabulary):

    '''
    returns the tf_idf_inverted_index file

    '''

    N = len(data)
    tf_idf_inverted_index = defaultdict(list)

    # document frequencies for each term in each description
    df = Counter()
    for d_id, description in data['cleaned_description'].items():
        terms = set(description)  # Get unique terms
        for t in terms:
            if t in vocabulary:
                df[t] += 1  #count documents containing this term
    
    # term frequencies for each document
    for d_id, description in data['cleaned_description'].items():
        tf = Counter(description) 
        tfidf_td = {}

        for t, tf_d in tf.items():
            if t in vocabulary:  # Check if term is in vocabulary
                t_id = vocabulary[t]
                df_td = df[t]    #document frequency of term t in doc d
                idf_t = math.log(N/df_td) if df_td > 0 else 0 #inverse doc frequency of term t
                tfidf= tf_d*idf_t  #TF-IDF 
                tfidf_td[t] = tfidf
                tf_idf_inverted_index[t_id].append((d_id, tfidf)) #using same ids as in vocabulary for consistency


    # Save to a JSON file
    with open('tf_idf_inverted_index.json', 'w') as file:
        json.dump(tf_idf_inverted_index, file, indent=4)

    return tf_idf_inverted_index




def query_vector(query, vocabulary, inverted_index):
    q = preprocessing(query)
    query_vector = {}

    #iterating over query terms as any non-query terms would have a TF of zero.
    for term in q:
        if term not in vocabulary:
            print(f"Term '{term}' not found in vocabulary")
            continue  # Skip terms not in the vocabulary

        term_id = vocabulary[term]

        tf = q.count(term)
        print(f"Term: {term}, TF: {tf}")

        if term_id in inverted_index.keys() and len(inverted_index[term_id]) > 0:
            df = len(inverted_index[term_id])
            idf = math.log( N/df )
        else:
            idf = math.log(N) #if it appears in 0 documents, term is very rare so idf is high

        print(f"Term: {term}, IDF: {idf}")

        tfidf = tf * idf
        print(f"Term: {term}, TF-IDF: {tfidf}")

        query_vector[term_id] = tfidf

    final_vector = [query_vector.get(term_id, 0) for term_id in vocabulary.values()]
    print(f"Final Query Vector: {final_vector}")
    return final_vector


def doc_vectors(data, vocabulary, tf_idf_inverted_index):
    
    num_docs = len(data)  # number of documents
    num_terms = len(vocabulary)  # number of terms

    # Initialize the document-term matrix (D), where rows correspond to terms and columns to documents
    D = np.zeros((num_terms, num_docs))

    # Iterate through each term in the inverted index
    for term_id, doc_scores in tf_idf_inverted_index.items():
        # Adjust term_id because vocabulary indices are off by 1 (due to header row)
        term_id_int = int(term_id) - 1  # term_id starts from 1 in vocabulary

        # Ensure the term_id is within the range of the number of terms
        if term_id_int < 0 or term_id_int >= num_terms:
            print(f"Skipping term_id {term_id_int} because it is out of bounds.")
            continue

        # Populate the document vector matrix for each document containing this term
        for doc_id, score in doc_scores:
            # Ensure the doc_id is within the range of the number of documents
            if doc_id < 0 or doc_id >= num_docs:
                print(f"Skipping doc_id {doc_id} because it is out of bounds.")
                continue

            # Assign the TF-IDF score to the appropriate position in the matrix
            D[term_id_int, doc_id] = score

    return D
