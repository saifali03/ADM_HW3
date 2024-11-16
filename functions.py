import requests
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
from nltk.stem import PorterStemmer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.stem import PorterStemmer, WordNetLemmatizer

def preprocess(text):
    """
    Preprocesses the input text by tokenizing it, removing stop words, and applying stemming and lemmatization.

    Args:
    text (str): The text to be preprocessed.

    Returns:
    list: The preprocessed text as a list of tokens.
    """

    # Tokenize the text
    tokens = word_tokenize(text)
    
    # Remove stop words and non-alphabetic tokens
    stop_words = set(stopwords.words('english'))
    tokens = [word.lower() for word in tokens if word.isalpha() and word.lower() not in stop_words]
    
    # Apply stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Apply lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    return tokens


def get_vocab(data, export_to_csv=False, csv_filename='vocabulary.csv'):
    """
    Builds a vocabulary from the cleaned descriptions in the input data.
    
    Args:
    data (pandas.DataFrame): The input data.
    
    export_to_csv (bool): Whether to export the vocabulary to a CSV file.
    
    csv_filename (str): The filename for the CSV file.
    
    Returns:
    dict: The vocabulary as a dictionary with words as keys and IDs as values.
    """
    vocab = {}
    
    # Iterate over cleaned descriptions to build vocabulary
    for description in data['description_clean']:
        for word in description:
            # Add word to vocab if not present, with the current vocab size as ID
            vocab.setdefault(word, len(vocab))
    
    # Optionally write vocab to a CSV file
    if export_to_csv:
        with open(csv_filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["term", "id"])  # Header row
            writer.writerows(vocab.items())
    
    return vocab

def search_engine(query, inverted_index, dataset, columns_for_dataset):
    
    """
    Steps:
    example query: "modern seasonal cuisine"
    1. Preprocess the query
    2. Intersect the list of document_ids the query words appear in using the following method:
       a. Pick the list of document_ids with the shortest length and intersect it with the second smallest
          list of document_ids. This can be done by sorting the query terms according to the length of
          their list of document_ids.
       b. Continue intersecting with the next smallest list until all lists are processed.
    3. What is left after the intersection is the list of document_ids to be returned.
    4. Use the returned list of doc ids to make a pandas DataFrame.
    5. Return the DataFrame.
    
    Args:
    query (str): The query to search for.
    inverted_index (dict): The inverted index to use for the search.
    dataset (pandas.DataFrame): The dataset to search in.
    columns_for_dataset (list): The columns in the dataset to include in the results.

    Returns:
    pandas.DataFrame: The search results as a pandas DataFrame. 
    
    """
    
    # Preprocess the query
    query_terms = preprocess(query)
    
    # Collect document lists for each query term if it exists in the inverted index
    doc_lists = [set(inverted_index[term]) for term in query_terms if term in inverted_index]
    
    # Check if there are terms with documents to process
    if doc_lists:
        # Sort document lists by their length (smallest to largest)
        doc_lists.sort(key=len)
        
        # Initialize docs_to_be_returned with the smallest list
        docs_to_be_returned = doc_lists[0]
        
        # Intersect with each subsequent list
        for doc_list in doc_lists[1:]:
            docs_to_be_returned.intersection_update(doc_list)
    else:
        # If no terms are found, return an empty set
        docs_to_be_returned = set()
    
    # Use the returned list of doc ids to make a pandas DataFrame
    result_df = dataset[dataset.index.isin(docs_to_be_returned)][columns_for_dataset]
    
    # Return the DataFrame
    return result_df

def inverted_index_tfidf(inverted_index, dataset):
    """
    Term Frequency: TF of a term or word is the number of times the term appears in a document compared to the total
    number of words in the document. 
    Inverse Document Frequency: Number of documents in the corpus divided by the number of documents in the corpus 
    that contain the term.
    Source learndatasci.com/glossary/tf-idf-term-frequency-inverse-document-frequency/
    """
    inverted_tfidf_index = {} # a dict with key as term_id of the word and as value: a tuple(row_at_which_word_occured, tfidf)
    words = list(inverted_index.keys()) # getting all the words of the vocabulary
    N = len(dataset) # number of documents in the corpus
    # Precompute document frequencies for each term
    doc_frequencies = {word: len(inverted_index[word]) for word in words}
    for word in words:
        indices = inverted_index[word] # getting all the indices (rows of dataset) where the word occured
        list_of_tuples = [] # to store list of tuple (row_at_which_word_occured, tfidf)
        for index in indices:
            dataset_row = dataset.at[index, "description_clean"] # going at that row
            term_count = dataset_row.count(word)
            tf = term_count / len(dataset_row) # computing the term frequency
            idf = math.log(N/doc_frequencies[word]) # computing the inverse doc frequency
            tf_idf = round(tf * idf, 2) # computing the tf * idf
            list_of_tuples.append((index,tf_idf)) # appending to the list ot tuples
        inverted_tfidf_index[word] = list_of_tuples # filling the index
    return inverted_tfidf_index

def calculate_query_tfidf(processed_query, inverted_index, data):
    """
    Calculate the TF-IDF for each term in the query.
    
    Args:
    processed_query (list): The preprocessed query terms.
    inverted_index (dict): The inverted index of the dataset.
    data (pandas.DataFrame): The dataset used to calculate the TF-IDF.
    
    Returns:
    dict: A dictionary with query terms as keys and their TF-IDF scores as values.
    """
    query_tfidf = {} # a dict with key as the word and as value the tfidf
    for word in processed_query: # for each word in the query
        if word in inverted_index: # if the word is in the vocabulary
            indices = inverted_index[word] # similar arguments as the inverted_index_tfidf
            tf = processed_query.count(word) / len(processed_query) # computing the term frequency
            idf = math.log(len(data) / len(indices)) # computing the inverse doc frequency
            tf_idf = round(tf * idf, 2) # computing the tf * idf
            query_tfidf[word] = tf_idf  # filling the dict
    return query_tfidf # returning the dict

# Calculate cosine similarity between the query TF-IDF vector and document TF-IDF vector
def cosine_similarity(query_tfidf, doc_tfidf):
    """
    Calculate the cosine similarity between the query TF-IDF vector and a document TF-IDF vector.
    
    Args:
    query_tfidf (dict): The TF-IDF vector for the query.
    doc_tfidf (dict): The TF-IDF vector for the document.
    
    Returns:
    float: The cosine similarity score.
    """
    # Calculate dot product
    dot_product = sum(query_tfidf[term] * doc_tfidf.get(term, 0) for term in query_tfidf)
    # Calculate magnitudes
    query_magnitude = math.sqrt(sum(value**2 for value in query_tfidf.values()))
    doc_magnitude = math.sqrt(sum(value**2 for value in doc_tfidf.values()))
    # Return cosine similarity
    if query_magnitude * doc_magnitude == 0:
        return 0
    return dot_product / (query_magnitude * doc_magnitude)


def search_engine_tfidf(inverted_tf_idf_index, query, inverted_index, data, columns_for_dataset):
    """
    Search engine using the TF-IDF and cosine similarity scores for ranking documents.

    Args:
    inverted_tf_idf_index (dict): The inverted index with TF-IDF scores.
    query (str): The query to search for.
    inverted_index (dict): The inverted index of the dataset.
    data (pandas.DataFrame): The dataset to search in.
    columns_for_dataset (list): The columns in the dataset to include in the results.

    Returns:
    pandas.DataFrame: The search results as a pandas DataFrame.
    """
    # a dict with key as the doc_id and as value a dict with key as the word and as value the tfidf
    doc_scores = defaultdict(dict)
    # Preprocess the query
    processed_query = preprocess(query)
    # Calculate the TF-IDF for each term in the query
    query_tfidf = calculate_query_tfidf(processed_query, inverted_index, data)
    # For each query term, gather relevant document vectors
    for term, _ in query_tfidf.items(): # for each term in the query
        if term in inverted_tf_idf_index: # if the term is in the inverted index
            # for each doc_id and doc_score in the inverted index of the term, where doc_score is the tfidf
            for doc_id, doc_tfidf in inverted_tf_idf_index[term]: 
                # fill the doc_scores dict.
                doc_scores[doc_id][term] = doc_tfidf

    # Calculate cosine similarity for each document
    cosine_similarities = {}
    # for each doc_id and doc_vector in the doc_scores
    for doc_id, doc_vector in doc_scores.items():
        # computing the cosine similarity between the query and the document vector
        cosine_similarities[doc_id] = cosine_similarity(query_tfidf, doc_vector)

    # Sort documents by similarity score
    ranked_docs = sorted(cosine_similarities.items(), key=lambda x: x[1], reverse=True)
    # get the indices of the documents in the dataset
    indices = [doc_id for doc_id, _ in ranked_docs]
    
    # Return the search results using the data and specified columns
    search_results = data[data.index.isin(indices)][columns_for_dataset]
    search_results['cosine_similarity'] = [cosine_similarities[doc_id] for doc_id in indices]
    search_results.reset_index(drop=True, inplace=True)
    return search_results

# Score adjustment functions
def get_all_cuisine_types(data):
    """
    Extracts all unique cuisine types from the dataset.
    
    Args:
    data (pandas.DataFrame): The dataset.
    
    Returns:
    set: A set of unique cuisine types.
    """
    # Initialize an empty set to store cuisine types
    cuisine_types = set()
    # Iterate over the 'cuisineType' column and update the set
    for cuisine in data['cuisineType']:
        # Split the string by ', ' and update the set
        cuisine_types.update(cuisine.split(', '))
    cuisine_types = {cuisine.lower() for cuisine in cuisine_types}
    # Return the set of unique cuisine types
    return cuisine_types

def get_all_facilities(data):
    """
    Extracts all unique facilities and services from the dataset.
    
    Args:
    data (pandas.DataFrame): The dataset.
    
    Returns:
    set: A set of unique facilities and services.
    """
    # Initialize an empty set to store facilities and services
    facilities = set()
    # Iterate over the 'facilitiesServices' column and update the set
    for facility in data['facilitiesServices']:
        # Split the string by '; ' and update the set
        facilities.update(facility.split('; '))
    facilities = {facility.lower() for facility in facilities}
    # Return the set of unique facilities and services
    return facilities

def adjust_score_for_price(raw_scores, data, synonyms_for_low_price, query_terms, doc_id):
    """
    Adjusts the score for documents with terms related to low prices.
    
    Args:
    raw_scores (dict): A dictionary of raw scores for documents.
    data (pandas.DataFrame): The dataset.
    synonyms_for_low_price (set): A set of synonyms for low prices.
    query_terms (list): A list of query terms.
    doc_id (int): The document ID.
    
    Returns:
    None"""
    # Check if any query term is related to low prices
    if any(term in synonyms_for_low_price for term in query_terms):
        # Check if any description term is related to low prices
        if any(word in synonyms_for_low_price for word in re.split(r'[,\s]+', str(data.loc[doc_id, 'description']))):
            raw_scores[doc_id] *= 1.2  # increase by 20%

def adjust_score_for_cuisine(raw_scores, cuisine_types, query_terms, doc_id):
    """
    Adjusts the score for documents with cuisine types related to the query.
    
    Args:
    raw_scores (dict): A dictionary of raw scores for documents.
    cuisine_types (set): A set of unique cuisine types.
    query_terms (list): A list of query terms.
    doc_id (int): The document ID.

    Returns:
    None
    """
    if any(any(term in cuisine for cuisine in cuisine_types) for term in query_terms):
        raw_scores[doc_id] *= 1.5  # increase by 50%

def adjust_score_for_facilities(raw_scores, facilities, query_terms, doc_id):
    """
    Adjusts the score for documents with facilities related to the query.
    
    Args:
    raw_scores (dict): A dictionary of raw scores for documents.
    facilities (set): A set of unique facilities and services.
    query_terms (list): A list of query terms.
    doc_id (int): The document ID.
    
    Returns:
    None
    """
    if any(any(term in facility for facility in facilities) for term in query_terms):
        raw_scores[doc_id] *= 1.3  # increase by 30%

def my_engine(query, search_results_tf_idf, data, synonyms_for_low_price, columns_for_dataset, facilities, cuisine_types, k=10):
    """
    Custom search engine with additional score adjustments

    Args:
    query: str, search query
    docs: list, list of document IDs that are relevant to the query
    search_results_tf_idf: pd.DataFrame, search results from the TF-IDF search engine
    data: pd.DataFrame, dataset
    synonyms_for_low_price: set, synonyms for low price
    columns_for_dataset: list, columns to include in the search results
    facilities: set, all facilities and services
    cuisine_types: set, all cuisine types
    k: int, number of search results to return

    Returns:
    pd.DataFrame, search results
    """

    # Calculate cosine similarity for each document only once
    raw_scores = search_results_tf_idf["cosine_similarity"].to_dict()

    # Apply additional score adjustments
    query_terms = [term.lower() for term in re.split(r'[,\s]+', query)]
    for doc_id in raw_scores:
        adjust_score_for_price(raw_scores, data, synonyms_for_low_price, query_terms, doc_id)
        adjust_score_for_cuisine(raw_scores, cuisine_types, query_terms, doc_id)
        adjust_score_for_facilities(raw_scores, facilities, query_terms, doc_id)

    # Use nlargest to get top k documents by score
    top_k_docs = nlargest(k, raw_scores.items(), key=lambda x: x[1])

    # Extract document IDs for the top k documents
    top_k_indices = [doc_id for doc_id, _ in top_k_docs]

    # Prepare the search results DataFrame
    search_results = data[data.index.isin(top_k_indices)][columns_for_dataset]
    search_results['similarity_score'] = [raw_scores[doc_id] for doc_id in top_k_indices]
    search_results.reset_index(drop=True, inplace=True)
    
    return search_results

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

def make_map(dataset, output_filename):
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
    for _, row in dataset.iterrows():
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
    