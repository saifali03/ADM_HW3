import nltk
import string
from constants import *
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

nltk.download('stopwords')
nltk.download('punkt')


stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text, stop_words=stop_words, stemmer=stemmer, tokenized=False):
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    
    # Tokenize the text 
    tokens = word_tokenize(text)
    
    # Remove stopwords
    tokens = [token for token in tokens if token not in stop_words]
    
    # Apply stemming
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    
    if tokenized:
        return stemmed_tokens
    else:
        # Join tokens back into a single string
        cleaned_text = " ".join(stemmed_tokens)
        return cleaned_text
    
    
def create_vocab_indexer(df, column):
    # Initialize an empty set to store unique tokens
    vocab = set()
    
    # Convert the specified column of the DataFrame to a NumPy array
    corpus = df[column].to_numpy()
    
    # Tokenize each text in the corpus and update the vocabulary set
    for text in corpus:
        tokens = word_tokenize(text)
        vocab.update(tokens)
    
    # Convert the vocabulary set into a DataFrame with a 'token' column
    vocab_df = pd.DataFrame(vocab, columns=['token'])  
    
    # Return the DataFrame containing unique tokens
    return vocab_df



def create_inverted_indexer(vocab_df, restaurants_df):
    # Initialize an empty dictionary to store the inverted index
    vocab_documents_dict = dict()
    
    # Iterate over each token in the vocabulary DataFrame
    for i, vocab in tqdm(enumerate(vocab_df.to_numpy())):
        # Find indices of rows in the DataFrame where the token appears in the 'cleaned_description' column
        indices = list(restaurants_df[restaurants_df['cleaned_description'].str.contains(vocab[0])].index)
        
        # Map the token's index in the vocabulary to the list of document indices
        vocab_documents_dict[i] = indices
    
    # Return the inverted index dictionary
    return vocab_documents_dict



def conjuctive_search(query, vocab_df, vocab_documents_dict, restaurants_df):
    # Clean and tokenize the query string
    tokens = clean_text(query, tokenized=True)
    
    # Get the document indices for the first token
    term_id = vocab_df[vocab_df.token == tokens[0]].index[0]
    common_rests = set(vocab_documents_dict[term_id])
    
    # Perform an intersection of document indices for all subsequent tokens
    for token in tokens[1:]:
        term_id = vocab_df[vocab_df.token == token].index[0]
        common_rests = common_rests & set(vocab_documents_dict[term_id])
    
    # Convert the result to a list of common restaurant indices
    common_rests = list(common_rests)
    
    # Retrieve and return details of the matching restaurants
    return restaurants_df.iloc[common_rests][['restaurantName', 'address', 'description', 'website']]



def create_inverted_indexer_with_tfidf(vocab_df, restaurants_df):
    vocab_documents_dict = dict()
    
    for i, row in tqdm(vocab_df.iterrows(), total=len(vocab_df)):
        vocab = row[0] 
        tfidf_scores = []
        
        # Find documents containing the current vocab term
        doc_indices = restaurants_df[
            restaurants_df['cleaned_description'].str.contains(fr'\b{vocab}\b', regex=True)
        ].index
        
        for index in doc_indices:
            # Get the document description
            description = restaurants_df.loc[index, 'cleaned_description']
            doc_tokens = description.split(' ')
            
            # Calculate term frequency (TF)
            matching_tokens = [token for token in doc_tokens if token == vocab]
            tf = len(matching_tokens) / len(doc_tokens)
            
            # Calculate inverse document frequency (IDF)
            if len(doc_indices) > 0:
                idf = np.log(len(restaurants_df) / len(doc_indices))
            else:
                idf = 0
            
            # Append the TF-IDF score
            tfidf_scores.append(tf * idf if tf > 0 else 0)
        
        # Store results for this term
        vocab_documents_dict[i] = list(zip(doc_indices, tfidf_scores))
    
    return vocab_documents_dict



def cosine_similarity(vector, matrix):
    # Compute the L2 norm (magnitude) of the input vector
    vector_norm = np.linalg.norm(vector, axis=1, keepdims=True)
    
    # Compute the L2 norm (magnitude) of the matrix rows
    matrix_norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    
    # Calculate the dot product between the vector and the matrix rows
    dot_product = vector @ matrix.T
    
    # Compute cosine similarity by normalizing the dot product with magnitudes
    similarity = dot_product / (vector_norm * matrix_norm.T)
    
    return similarity  # Return the cosine similarity scores



def rank_search_engine(query, vocab_df, vocab_documents_with_tfidf, restaurants_df, k=5):
    # Step 1: Process the query
    tokens = clean_text(query, tokenized=True)
    vectorized_query = np.zeros(len(vocab_df))
    vocab_dict = {vocab: idx for idx, vocab in enumerate(vocab_df.to_numpy().flatten())}
    
    for token in tokens:
        if token in vocab_dict:
            vectorized_query[vocab_dict[token]] = 1
    
    vectorized_query = vectorized_query.reshape(1, -1)
    
    # Step 2: Construct document vectors
    restaurants_dict = {restaurant: np.zeros(len(vocab_df)) for restaurant in restaurants_df.index}
    for vocab_id, rest_tfidf in vocab_documents_with_tfidf.items():
        for restaurant, tfidf_score in rest_tfidf:
            restaurants_dict[restaurant][vocab_id] = tfidf_score
    
    restaurant_ids = list(restaurants_dict.keys())
    tfidf_matrix = np.array([restaurants_dict[id] for id in restaurant_ids])
    
    # Step 3: Compute cosine similarity and rank results
    similarity_scores = cosine_similarity(vectorized_query, tfidf_matrix).flatten()
    top_k_restaurants = np.argsort(similarity_scores)[::-1][:k]
    
    # Step 4: Return results
    selected_rests = restaurants_df.iloc[top_k_restaurants]
    selected_rests['Similarity Score'] = np.sort(similarity_scores)[::-1][:k]
    return selected_rests


