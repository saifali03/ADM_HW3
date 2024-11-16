
import pandas as pd 
import numpy as np 


# Function to convert price range to numeric based on the count of '€' symbols
def convert_to_numeric(value):
    if pd.isna(value):  # If the value is NaN, leave it as NaN
        return np.nan
    elif isinstance(value, str) and '€' in value:  # If the string contains '€' symbol(s)
        return value.count('€')  # Count the number of '€' symbols
    return np.nan  # If no '€' symbols, return NaN


def conjuntive_query(query, dictionary, inverted_index):

    #preprocessing query text with the same function used to clean descriptions
    cleaned_query = preprocessing(query)

    unique_words = set(cleaned_query)

    doc_lists = []
    for word in unique_words:
        #retrive the corresponding term_id from the dictionary
        term_id = dictionary.get(word)
        if term_id:
            #retrive and append the list of documents containing term_id
            docs = inverted_index.get(str(term_id))
            if list:
                doc_lists.append(docs)
    
    if doc_lists:
        intersection = doc_lists[0] #initialize as the first list
        for docs in doc_lists[1:]:
            intersection = list(set(intersection) & set(docs)) #iteratively intersect with any other list
    else:
        intersection = []

    result = data.loc[intersection, ['restaurantName', 'address', 'description', 'website','priceRange', 'cuisineType', 'facilitiesServices','id']]
    print(query)

    if intersection:
        return result
    else:
        print('No matches found')
        return None


def customized_query(query_text, dictionary, inverted_index, TFiDF, vocabulary, query_price, query_services):
    filter_precise_match = conjuntive_query(query_text, dictionary, inverted_index)
    filter_precise_match['priceRange_numeric'] = filter_precise_match['priceRange'].apply(convert_to_numeric)
    filter_precise_match['score'] = 0
    for w in query_text.split(' '):
        for index, ristorante in filter_precise_match.iterrows(): 
            if w in vocabulary: 
                TFIDF_w = {str(elemento[0]): elemento[1] for elemento in TFiDF[str(vocabulary[w])]}  
                id_ristorante = ristorante['id'] 
                if id_ristorante:  
                    TFIDF_w_r = TFIDF_w.get(str(id_ristorante), 0)  
                    filter_precise_match.at[index, 'score'] += TFIDF_w_r 
    filter_precise_match.sort_values(by='score', ascending=False, inplace=True)
    
    for i, restaurant in filter_precise_match.iterrows():
        price_value = restaurant['priceRange_numeric']
        price_diff = price_value - query_price
        score_price = -0.5 * price_diff
        if price_diff == 0:
            score_price = 2
        filter_precise_match.at[i, 'score'] += score_price

        if isinstance(restaurant['facilitiesServices'], str):
            score_services = sum(1 for service in query_services if service in restaurant['facilitiesServices'].split(';')) * 0.1
        else:
            score_services = 0

        if isinstance(restaurant['cuisineType'], str):
            score_cuisine = sum(1 for service in query_services if service in restaurant['cuisineType'].split(',')) * 0.1
        else:
            score_cuisine = 0

        filter_precise_match.at[i, 'score'] += score_price + score_cuisine + score_services

    filter_precise_match.sort_values(by='score', ascending=False, inplace=True)
    
    return filter_precise_match
