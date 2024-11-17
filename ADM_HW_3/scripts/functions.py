import requests
import pandas as pd
import folium
from folium.plugins import MarkerCluster
import re
import math
from collections import defaultdict
from heapq import nlargest
import csv
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
import heapq
import json
import numpy as np

stop_words = set(stopwords.words('english')) 
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

def preprocessing(string):
    '''
    Preprocesses the input text by applying the following steps:
    1. Converts the text to lowercase.
    2. Removes extra whitespace and replaces hyphens with spaces to handle compound words.
    3. Tokenizes the text into individual words.
    4. Removes punctuation, retaining only alphanumeric tokens.
    5. Filters out stopwords to reduce noise.
    6. Applies stemming to reduce words to their root form, optimizing vocabulary size and search efficiency.

    Parameters:
    string (str): The input text to be preprocessed.

    Returns:
    list: A list of preprocessed tokens ready for further processing or analysis.
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


def create_vocabulary(descriptions):
    '''
    Builds a vocabulary from the input descriptions by assigning a unique term ID to each distinct term.
    Writes the resulting vocabulary to a CSV file.

    Steps:
    1. Initializes an empty vocabulary dictionary and term IDs starting at 1.
    2. Iterates through each description and extracts unique words (using a set to avoid duplicates).
    3. Adds terms to the vocabulary if they are digits or have more than one character.
    4. Saves the vocabulary to a CSV file named 'vocabulary.csv', with columns 'term' and 'term_id'.

    Parameters:
    descriptions (list of lists): A list of preprocessed descriptions, where each description is a list of words.

    Returns:
    dict: A dictionary mapping terms to their unique term IDs.
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
    with open('Files/vocabulary.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['term', 'term_id'])
        for term, term_id in vocabulary.items():
            writer.writerow([term, term_id])
    
    return vocabulary



def create_inverted_index(descriptions, vocabulary):
    '''
    Creates an inverted index from the given descriptions and vocabulary, and saves it as a JSON file.

    Steps:
    1. Initializes an empty `defaultdict` for the inverted index, where each term ID maps to a list of document IDs.
    2. Iterates over each description (document) with its corresponding `doc_id`.
    3. Extracts unique words from the description to avoid duplicate entries for the same term in the same document.
    4. Maps each word to its term ID using the vocabulary and updates the inverted index.
    5. Saves the resulting inverted index to a file named 'inverted_index.json'.

    Parameters:
    descriptions (list of lists): A list where each element is a preprocessed description (list of words).
    vocabulary (dict): A dictionary mapping terms to their unique term IDs.

    Returns:
    defaultdict: An inverted index mapping term IDs to a list of document IDs where they appear.
    '''

    inverted_index = defaultdict(list)
    
    # Iterate over descriptions (documents) were doc_id is the row index (uniques)
    for doc_id, description in enumerate(descriptions, start=0):

        unique_words = set(description)  # Remove duplicates

        for word in unique_words:
            if word in vocabulary:
                term_id = vocabulary[word]   #map the term to the corresponding document ID
                if doc_id not in inverted_index[term_id]: # to avoid duplicated for the same term in the same document
                    inverted_index[term_id].append(doc_id)
    
    # Save the inverted index
    with open('Files/inverted_index.json', 'w') as file:
        json.dump(inverted_index, file, indent=4)

    return inverted_index


def load_dictionary(csv_file):
    '''
    Loads a term-to-term_id mapping from a CSV file.

    Steps:
    1. Opens the specified CSV file.
    2. Reads each row (skipping the header).
    3. Stores each term and its corresponding term_id in a dictionary.

    Parameters:
    csv_file (str): Path to the CSV file containing the vocabulary.

    Returns:
    dict: A dictionary mapping terms to their respective term_ids.
    '''
    dictionary = {}
    with open(csv_file, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        for row in reader:
            term, term_id = row
            dictionary[term] = int(term_id)  # Store term and its term_id
    return dictionary

def load_inverted_index(json_file):

    '''
    Loads an inverted index from a JSON file.

    Steps:
    1. Opens the specified JSON file.
    2. Reads the contents of the file into a dictionary.

    Parameters:
    json_file (str): Path to the JSON file containing the inverted index.

    Returns:
    dict: An inverted index mapping term_ids to lists of document IDs.
    '''
    with open(json_file, 'r') as file:
        inverted_index = json.load(file)
    return inverted_index


def highlight_matches(result_df, query_words):
    '''
    Highlights query matches in the description field of the search results and prints the relevant restaurant details.

    Steps:
    1. Iterates through each row in the result DataFrame containing search results.
    2. Identifies the query words that appear in the description of each restaurant.
    3. Highlights the matched words in the description by surrounding them with '**'.
    4. Prints the restaurant's name, address, description (with highlighted matches), and website.

    Parameters:
    result_df (DataFrame): A DataFrame containing restaurant details including name, address, description, and website.
    query_words (list of str): A list of query terms to search for in the descriptions.

    Returns:
    None: Prints the formatted output with highlighted matches.
    '''

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
    This function generates a TF-IDF inverted index for a given dataset of cleaned restaurant descriptions.
    It calculates the term frequency (TF) and inverse document frequency (IDF) for each term and 
    stores the results in an inverted index format.

    Steps:
    1. Calculate the document frequency (DF) for each term in the vocabulary.
    2. Calculate the term frequency (TF) for each document's cleaned description.
    3. Calculate the TF-IDF score for each term in each document.
    4. Store the results in an inverted index, where each term (from the vocabulary) is mapped to a 
       list of document IDs and their corresponding TF-IDF scores.
    5. Save the resulting inverted index as a JSON file ('tf_idf_inverted_index.json').

    Input parameters:
    - data (pandas DataFrame): A DataFrame containing the 'cleaned_description' column, where each row 
      represents a document (restaurant description) with cleaned and preprocessed text.
    - vocabulary (dict): A dictionary mapping terms (words) to unique term IDs. This helps ensure consistency 
      in indexing and prevents using terms that are not part of the vocabulary.

    Output:
    - tf_idf_inverted_index (defaultdict): A dictionary where the keys are term IDs (from the vocabulary), 
      and the values are lists of tuples. Each tuple contains a document ID and the corresponding TF-IDF score 
      for that term in that document.
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
    with open('Files/tf_idf_inverted_index.json', 'w') as file:
        json.dump(tf_idf_inverted_index, file, indent=4)

    return tf_idf_inverted_index


def query_vector(query, vocabulary, inverted_index, data):
    '''
    This function generates a query vector for a given query string. It computes the TF-IDF 
    score for each term in the query based on its frequency in the query and its inverse 
    document frequency in the dataset. The resulting vector is created with the same structure 
    as the document vectors, using the vocabulary and inverted index.

    Steps:
    1. Preprocess the query string to clean and normalize the terms.
    2. For each term in the query, check if it exists in the vocabulary.
    3. Calculate the term frequency (TF) of each term in the query.
    4. Calculate the inverse document frequency (IDF) for each term using the inverted index.
    5. Calculate the TF-IDF score for each term by multiplying TF and IDF.
    6. Construct the final query vector where each term is represented by its TF-IDF score.
    7. Return the query vector as a list of TF-IDF values corresponding to terms in the vocabulary.

    Input parameters:
    - query (str): The query string to be processed. This string contains the terms for which 
      the query vector will be computed.
    - vocabulary (dict): A dictionary that maps terms (words) to unique term IDs. It is used 
      to ensure consistency in indexing.
    - inverted_index (dict): A dictionary where the keys are term IDs and the values are lists 
      of tuples, each containing a document ID and the corresponding TF-IDF score for that term.
    - data (DataFrame) : Our dataset

    Output:
    - final_vector (list): A list of TF-IDF values representing the query vector. The length of 
      the list corresponds to the number of terms in the vocabulary, and the values correspond to 
      the TF-IDF scores of each term in the query. If a term in the vocabulary is not in the query, 
      its corresponding TF-IDF score will be 0.
    '''
    q = preprocessing(query)
    query_vector = {}
    N = len(data)
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
    '''
    This function generates a document-term matrix (D) using the TF-IDF scores from an inverted index. 
    Each row of the matrix corresponds to a term from the vocabulary, and each column corresponds to a document. 
    The matrix is populated with TF-IDF values, where each entry represents the importance of a term in a document.

    Steps:
    1. Initialize an empty document-term matrix (D) with dimensions (num_terms x num_docs).
    2. Iterate through the inverted index to retrieve the term IDs and their corresponding TF-IDF scores for each document.
    3. For each term, assign its TF-IDF score to the corresponding position in the matrix.
    4. Ensure that the term ID and document ID are within valid ranges and update the matrix accordingly.
    5. Return the document-term matrix populated with TF-IDF scores.

    Input parameters:
    - data (pandas DataFrame): A DataFrame containing document data. Each row represents a document, and the number of rows corresponds to the number of documents.
    - vocabulary (dict): A dictionary mapping terms (words) to unique term IDs. This ensures consistency in indexing terms for the matrix.
    - tf_idf_inverted_index (dict): A dictionary where the keys are term IDs and the values are lists of tuples. Each tuple contains a document ID and the corresponding TF-IDF score for that term.

    Output:
    - D (numpy array): A 2D NumPy array representing the document-term matrix. Each entry D[i][j] contains the TF-IDF score for term i in document j. 
      If a term does not appear in a document, its corresponding entry will be 0.
    '''

    
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


def cosine_similarity(a,b):
    cos_sim = np.dot(a,b) / (np.linalg.norm(a)*np.linalg.norm(b))
    return cos_sim


def ranked_search(query, k, vocabulary, data, inverted_index, D):
    '''
    This function performs a ranked search to retrieve the top-k most similar documents to a given query. 
    It calculates the cosine similarity between the query vector and each document vector in the document-term matrix 
    and returns the top-k documents based on their similarity scores.

    Steps:
    1. Convert the input query into a query vector using the `query_vector` function.
    2. Calculate the cosine similarity between the query vector and each document vector in the document-term matrix.
    3. Store the similarity scores for each document.
    4. Sort the documents based on their similarity scores in descending order.
    5. Return the top-k documents with the highest similarity scores.

    Input parameters:
    - query (str): The query string to search for in the dataset. This string will be processed and converted to a query vector.
    - k (int): The number of top results to return based on the highest similarity scores.
    - vocabulary (dict): The vocabulary mapping terms to term IDs.
    - data (pandas DataFrame): The dataset containing documents and associated information.
    - inverted_index (dict): The inverted index for the documents.
    - D (numpy array): The document-term matrix where each column corresponds to a document's vector.


    Output:
    - top_k_data (pandas DataFrame): A DataFrame containing the top-k documents sorted by similarity score. 
      It includes the restaurant name, address, description, website, and the similarity score.
    '''
    
    q = query_vector(query, vocabulary, inverted_index, data)
    similarity_scores = []
   
    for doc_id in range(len(data)):
        # document vector (doc_id-th column of V)
        doc_vector = D[:, doc_id]
        s = cosine_similarity(q, doc_vector)
        similarity_scores.append(s)

    data['SimilarityScore'] = similarity_scores

    #sorting rows in dataset
    top_k_data = data.sort_values(by='SimilarityScore', ascending=False).head(k)[['restaurantName', 'address', 'description', 'website', 'SimilarityScore']]
   
   
    return top_k_data


def conjunctive_query(query, dictionary, inverted_index, data):
    """
    Perform a conjunctive query to find matching documents based on the query terms.
    
    Args:
        query (str): The user query text.
        dictionary (dict): A mapping of terms to their term IDs.
        inverted_index (dict): A mapping of term IDs to lists of document IDs.
        data (pd.DataFrame): A DataFrame containing restaurant information.
    Returns:
        pd.DataFrame: A DataFrame containing matching documents with specific fields, or None if no matches are found.
    """
    # Preprocess the query text
    cleaned_query = preprocessing(query)
    unique_words = set(cleaned_query)

    doc_lists = []
    for word in unique_words:
        # Retrieve term ID from the dictionary
        term_id = dictionary.get(word)
        if term_id:
            # Retrieve the document list for the term
            docs = inverted_index.get(str(term_id))
            if docs:
                doc_lists.append(docs)

    if doc_lists:
        # Compute the intersection of document lists
        intersection = doc_lists[0]
        for docs in doc_lists[1:]:
            # Use heapq for efficient intersection computation
            intersection = list(heapq.nsmallest(len(intersection), set(intersection) & set(docs)))
    else:
        intersection = []

    # Retrieve relevant data from the DataFrame
    if intersection:
        result = data.loc[intersection, [
            'restaurantName', 'address', 'description', 'website',
            'priceRange', 'cuisineType', 'facilitiesServices', 'id'
        ]]
        return result
    else:
        print('No matches found')
        return None

def convert_to_numeric(price_range):
    """
    Convert a price range string to a numeric value.
    
    Args:
        price_range (str): A price range string.
        
    Returns:
        float: The average price value.
    """
    if price_range == '€':
        return 1
    elif price_range == '€€':
        return 2
    elif price_range == '€€':
        return 3
    elif price_range == '€€€':
        return 4
    elif price_range == '€€€€':
        return 5
    else:
        return 0

def conjunctive_query_luca(query, dictionary, inverted_index, data):
    """
    Perform a conjunctive query to find matching documents based on the query terms.
    
    Args:
        query (str): The user query text.
        dictionary (dict): A mapping of terms to their term IDs.
        inverted_index (dict): A mapping of term IDs to lists of document IDs.
        data (pd.DataFrame): A DataFrame containing restaurant information.
    Returns:
        pd.DataFrame: A DataFrame containing matching documents with specific fields, or None if no matches are found.
    """
    # Preprocess the query text
    cleaned_query = preprocessing(query)
    unique_words = set(cleaned_query)

    doc_lists = []
    for word in unique_words:
        # Retrieve term ID from the dictionary
        term_id = dictionary.get(word)
        if term_id:
            # Retrieve the document list for the term
            docs = inverted_index.get(str(term_id))
            if docs:
                doc_lists.append(docs)

    if doc_lists:
        # Compute the intersection of document lists
        intersection = doc_lists[0]
        for docs in doc_lists[1:]:
            # Use heapq for efficient intersection computation
            intersection = list(heapq.nsmallest(len(intersection), set(intersection) & set(docs)))
    else:
        intersection = []

    # Retrieve relevant data from the DataFrame
    if intersection:
        valid_indices = [doc_id for doc_id in intersection if doc_id in data.index]
        result = data.loc[valid_indices, [
            'restaurantName', 'address', 'description', 'website',
            'priceRange', 'cuisineType', 'facilitiesServices', 'id'
        ]]
        return result
    else:
        print('No matches found')
        return None
    
def customized_query(query_text, dictionary, inverted_index, TFiDF, vocabulary, query_price, query_services, data):
    """
    Perform a customized query to rank documents based on user preferences and scores.
    
    Args:
        query_text (str): The user query text.
        dictionary (dict): A mapping of terms to their term IDs.
        inverted_index (dict): A mapping of term IDs to lists of document IDs.
        TFiDF (dict): Term frequency-inverse document frequency scores.
        vocabulary (dict): A mapping of terms to their vocabulary indices.
        query_price (float): The user's desired price range.
        query_services (list): A list of desired facilities or services.
        
    Returns:
        pd.DataFrame: A DataFrame containing ranked restaurants with scores.
    """
    # Retrieve precise matches using the conjunctive query function
    filter_precise_match = conjunctive_query_luca(query_text, dictionary, inverted_index, data)
    if filter_precise_match is None or filter_precise_match.empty:
        return pd.DataFrame()  # Return an empty DataFrame if no matches are found

    # Convert price ranges to numeric values
    filter_precise_match['priceRange_numeric'] = filter_precise_match['priceRange'].apply(convert_to_numeric)
    filter_precise_match['score'] = 0  # Initialize scores

    # Compute TF-IDF scores for each query term
    for term in query_text.split():
        if term in vocabulary:
            term_id = vocabulary[term]
            term_tfidf = {str(entry[0]): entry[1] for entry in TFiDF.get(str(term_id), [])}
            for index, row in filter_precise_match.iterrows():
                doc_id = row['id']
                tfidf_score = term_tfidf.get(str(doc_id), 0)
                filter_precise_match.at[index, 'score'] += tfidf_score

    # Add additional scoring based on price, facilities, and cuisine types
    for i, row in filter_precise_match.iterrows():
        # Price score adjustment
        price_value = row['priceRange_numeric']
        price_diff = price_value - query_price
        price_score = -0.5 * price_diff if price_diff != 0 else 2

        # Services score adjustment
        if isinstance(row['facilitiesServices'], str):
            services_score = sum(1 for service in query_services if service in row['facilitiesServices'].split(';')) * 0.1
        else:
            services_score = 0

        # Cuisine type score adjustment
        if isinstance(row['cuisineType'], str):
            cuisine_score = sum(1 for service in query_services if service in row['cuisineType'].split(',')) * 0.1
        else:
            cuisine_score = 0

        # Update total score
        filter_precise_match.at[i, 'score'] += price_score + services_score + cuisine_score

    # Round scores for readability
    filter_precise_match['score'] = filter_precise_match['score'].round(2)

    # Rank the restaurants by score in descending order
    sorted_restaurants = heapq.nlargest(
        len(filter_precise_match),
        filter_precise_match.itertuples(index=False),
        key=lambda x: x.score
    )

    # Convert sorted results back to a DataFrame
    columns = ['restaurantName', 'address', 'description', 'website', 'score']
    result = pd.DataFrame.from_records(
        [(r.restaurantName, r.address, r.description, r.website, r.score) for r in sorted_restaurants],
        columns=columns
    )

    return result




def get_city(address):
    """
    Extracts the city from an address string.
    
    Args:
    address (str): The address string.
    
    Returns:
    str: The city name.
    """
    # Split the address string by commas and return the third last element - which is the city
    return address.split(",")[-3].strip()

def get_region_and_coordinates(city, API_KEY):
    """
    Extracts the region and coordinates (latitude, longitude) for a given city in Italy.
    
    Args:
    city (str): The city name.
    API_KEY (str): The Google Maps API key.
    
    Returns:
    tuple: A tuple containing the region and coordinates (latitude, longitude).
    """
    # Get geocode data for the city
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={city.replace(' ', '+')},+Italy&key={API_KEY}"
    # Get the response
    response = requests.get(geocode_url).json()
    
    # Extract region from address components
    region = None
    # Check if the response status is OK
    if response['status'] == 'OK':
        # Iterate over the address components
        for component in response['results'][0]['address_components']:
            # Check if the component type is 'administrative_area_level_1' - it is where the region is stored
            if 'administrative_area_level_1' in component['types']:
                # Extract the region name
                region = component['long_name']
                break
    
    # Extract coordinates given the region
    if region:
        # Get geocode data for the region
        region_geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={region.replace(' ', '+')},+Italy&key={API_KEY}"
        # Get the response
        region_response = requests.get(region_geocode_url).json()
        # Check if the response status is OK
        if region_response['status'] == 'OK':
            # Extract latitude and longitude
            lat = region_response['results'][0]['geometry']['location']['lat']
            lng = region_response['results'][0]['geometry']['location']['lng']
            # Return the region and coordinates
            return (region, lat, lng)
    return (None, None, None)

def get_coordinates(address, API_KEY):
    """
    Extracts the coordinates (latitude, longitude) for a given address.
    
    Args:
    address (str): The address string.
    API_KEY (str): The Google Maps API key.
    
    Returns:
    tuple: A tuple containing the latitude and longitude.
    """
    
    # Get geocode data for the address
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={address.replace(' ', '+')},&key={API_KEY}"
    
    # Get the response
    response = requests.get(geocode_url).json()
    # Extract coordinates if the response status is OK
    if response['status'] == 'OK':
        lat = response['results'][0]['geometry']['location']['lat']
        lng = response['results'][0]['geometry']['location']['lng']
        # Return the coordinates
        return (lat, lng)
    return (None, None)

def make_map(dataset, region_data, output_filename):
    """
    Creates a map with regions encircled and markers for each restaurant in the dataset.
    
    Args:
    dataset (pandas.DataFrame): The dataset containing restaurant information.
    output_filename (str): The filename to save the map.
    
    Returns:
    None
    """
    # Create a base map using Folium. location coordinates are for Rome (Central Italy)
    italy_map = folium.Map(location=[41.8719, 12.5674], zoom_start=6)

    # Add markers for each region
    for _, row in region_data.iterrows():
        # Add a circle around the region
        folium.Circle(
            # Use the region's latitude and longitude as the circle's center
            location=[row['lat_region'], row['lng_region']],
            radius=65000,  # Adjust radius as needed (in meters)
            color=row["color_region"],  # Circle border color
            fill=True, # Fill the circle with the fill color
            fill_color=row["color_region"],  # Fill color of the circle
            fill_opacity=0.01,  # the higher the value, the more opaque the circle
            popup=f"{row['region']}", # Popup text when clicking on the circle
            tooltip=f"{row['region']}", # Tooltip text when hovering over the circle
        ).add_to(italy_map)

    # Create a MarkerCluster object and add it to the map. It helps to cluster multiple markers together and render the plot neatly.
    marker_cluster = MarkerCluster().add_to(italy_map)

    # Add restaurant markers to the cluster
    for _, row in dataset.iterrows():
        # Add a marker for each restaurant
        folium.Marker(
            # Use the restaurant's latitude and longitude as the marker's location
            location=[row['address_lat'], row['address_lng']],
            # Add a popup with restaurant information
            popup=f"<b>Name:</b> {row['restaurantName']}<br><b>Price Range:</b> {row['priceRange']}<br><b>Cuisine:</b> {row['cuisineType']} <br><b>Facilities:</b> {row['facilitiesServices']} <br><b>website:</b> {row['website'] if row['website'] else None}",
            # Add a tooltip with the restaurant's name and cuisine type
            tooltip=f"Name: {row['restaurantName']} <br> Cuisine Type: {row['cuisineType']}",
            # Use a custom icon with the color based on the price range
            icon=folium.Icon(color=row["color"], icon="cutlery", prefix="fa")
        # Add the marker to the cluster
        ).add_to(marker_cluster)

    # Add a legend (custom HTML template) # generated with the help of Chat-GPT
    legend_html = """
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000; background-color: white; 
                padding: 10px; border: 2px solid black; border-radius: 5px;">
        <h4>Price Range Legend</h4>
        <i style="background: green; width: 10px; height: 10px; display: inline-block;"></i> €<br>
        <i style="background: yellow; width: 10px; height: 10px; display: inline-block;"></i> €€<br>
        <i style="background: orange; width: 10px; height: 10px; display: inline-block;"></i> €€€<br>
        <i style="background: red; width: 10px; height: 10px; display: inline-block;"></i> €€€€
    </div>
    """
    # Add the legend to the map
    italy_map.get_root().html.add_child(folium.Element(legend_html))

    # Save the map
    italy_map.save(output_filename)
    print(f"Map saved to {output_filename}")
    