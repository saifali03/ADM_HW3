# Import the relevant libraries
from bs4 import BeautifulSoup
import os
import re
import csv
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

# we will add docstrings to the functions

# Restaurant Name (to save as restaurantName): string with select method
def get_restaurant_name(soup) -> str:
    """
    Extract the restaurant name from the HTML content.

    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: string with the restaurant name or None if not found
    """
    try: # try to find the restaurant name
        restaurant_name = soup.find("h1", class_="data-sheet__title") # find the restaurant name
        return restaurant_name.get_text(strip=True) if restaurant_name else None # return the restaurant name if found
    except AttributeError:
        return None # return None if the restaurant name is not found

def get_address(soup) -> str:
    """
    Extract the address from the HTML content.

    param soup: BeautifulSoup object containing the parsed HTML content

    return: string with the address or None if not found
    """
    content = soup.find("div", class_="data-sheet__block--text") # find the address content
    return content.get_text(strip=True) if content else None # return the address if found

def get_city(address) -> str:
    """ 
    Extract the city from the address string.
    
    param address: string with the address information
    
    return: string with the city name or None if not found
    """
    if address: # check if the address is not None
        parts = address.split(",") # split the address by comma
        if len(parts) > 3: # check if the length of the parts is greater than 3
            return parts[-3].strip() # return the city name (third last part)
    return None # return None if the city is not found

# Postal Code extraction - like 00143
def get_postal_code(address) -> str:
    """
    Extract the postal code from the address string.
    
    param address: string with the address information
    
    return: string with the postal code or None if not found
    """
    postal_code = re.search(r"\b\d{5}\b", address) # search for a 5-digit number in the address
    return postal_code.group() if postal_code else None # return the postal code if found

# Country extraction
def get_country(address) -> str:
    """
    Extract the country from the address string.
    
    param address: string with the address information
    
    return: string with the country name or None if not found
    """
    if address: # check if the address is not None
        parts = address.split(",") # split the address by comma
        if len(parts) > 1: # check if the length of the parts is greater than 1
            return parts[-1].strip() # return the country name (last part)
    return None # return None if the country is not found

def get_price_range(soup) -> str:
    """
    Extract the price range from the HTML content.
    
    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: string with the price range or None if not found
    """
    try: # try to find the price range
        content_blocks = soup.find_all("div", class_="data-sheet__block--text") # find all the content blocks
        if len(content_blocks) > 1: # check if the length of the content blocks is greater than 1
            for block in content_blocks: # iterate over the content blocks
                if "€" in block.get_text(strip=True): # check if the block contains the euro symbol
                    price = re.findall("€", block.get_text(strip=True)) # find all the euro symbols
                    price_range = ''.join(price) # join the euro symbols
                    return price_range # return the price range
    except AttributeError: # handle the attribute error
        return None # return None if the price range is not found

def get_cuisine_type(soup) -> str:
    """
    Extract the cuisine type from the HTML content.
    
    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: string with the cuisine type or None if not found
    """
    try: # try to find the cuisine type
        content_blocks = soup.find_all("div", class_="data-sheet__block--text") # find all the content blocks
        if len(content_blocks) > 1: # check if the length of the content blocks is greater than 1
            for block in content_blocks: # iterate over the content blocks
                if "€" in block.get_text(strip=True): # check if the block contains the euro symbol
                    cuisine_info = block.get_text(strip=True).split("·")[1] # split the block by the bullet symbol
                    # the cuisine type is the second part after the bullet symbol
                    return cuisine_info.strip() # return the cuisine type
        cuisine_info = content_blocks[-1].get_text(strip=True) # get the last content block
        return cuisine_info # return the cuisine type
    except AttributeError:
        return None

def get_description(soup) -> str:
    """
    Extract the description from the HTML content.
    
    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: string with the description or None if not found
    """
    description_block = soup.find("div", class_="data-sheet__description") # find the description block
    return description_block.get_text(strip=True) if description_block else None # return the description if found

def get_facilities_services(soup) -> list:
    """
    Extract the facilities and services from the HTML content.
    
    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: list of strings with the facilities and services or an empty list if not found
    """
    services_section = soup.find("div", class_="restaurant-details__services") # find the services section
    if services_section: # check if the services section is found
        facilities = services_section.find_all("li") # find all the list items in the services section
        return [facility.get_text(strip=True) for facility in facilities] # return the text of each list item
    return [] # return an empty list if the services section is not found

def get_credit_cards(soup) -> list:
    """
    Extract the accepted credit cards from the HTML content.
    
    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: list of strings with the credit cards or an empty list if not found
    """
    card_section = soup.find("div", class_="list--card") # find the card section
    cards_list = [] # initialize an empty list to store the credit cards
    if card_section: # check if the card section is found
        cards = card_section.find_all("img") # find all the image tags in the card section
        pattern = r'(?<=/icons/)([^-]+)' # regex pattern to extract the card name
        for card in cards: # iterate over the cards
            src = card.get("data-src", "") # get the data-src attribute of the image tag. If not found, return an empty string
            match = re.search(pattern, src) # search for the card name using the regex pattern
            if match: # check if the match is found
                cards_list.append(match.group(1)) # append the card name to the list
    return cards_list # return the list of credit cards

def get_phone_number(soup) -> str:
    """
    Extract the phone number from the HTML content.
    
    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: string with the phone number or None if not found
    """
    content = soup.find("span", {"class": "flex-fill", "dir": "ltr", "x-ms-format-detection": "none"}) # find the phone number
    return content.get_text(strip=True) if content else None # return the phone number if found

def get_website(soup) -> str:
    """
    Extract the website URL from the HTML content.
    
    param soup: BeautifulSoup object containing the parsed HTML content
    
    return: string with the website URL or None if not found
    """
    content = soup.find("div", class_="collapse__block-item link-item") # find the block with the website URL
    if content: # check if the content is found
        website = content.find("a", href=True) # find the anchor tag with the website URL
        return website["href"] if website else None # return the website URL if found
    return None

def extract_restaurant_data(html_file_path) -> dict:
    """
    Extract restaurant data from an HTML file.
    
    param html_file_path: string with the path to the HTML file
    
    return: dictionary with the extracted data
    """
    with open(html_file_path, 'r', encoding='utf-8') as file: # open the HTML file
        soup = BeautifulSoup(file, 'lxml')  # Parse the HTML content of the file
        
        # Extract data using provided functions
        restaurant_name = get_restaurant_name(soup) 
        address = get_address(soup)
        city = get_city(address) if address else None
        postal_code = get_postal_code(address) if address else None
        country = get_country(address) if address else None
        price_range = get_price_range(soup)
        cuisine_type = get_cuisine_type(soup)
        description = get_description(soup)
        facilities_services = get_facilities_services(soup)
        credit_cards = get_credit_cards(soup)
        phone_number = get_phone_number(soup)
        website = get_website(soup)
        
        # Return data as a dictionary
        return {
            "restaurantName": restaurant_name,
            "address": address,
            "city": city,
            "postalCode": postal_code,
            "country": country,
            "priceRange": price_range,
            "cuisineType": cuisine_type,
            "description": description,
            "facilitiesServices": "; ".join(facilities_services),
            "creditCards": "; ".join(credit_cards),
            "phoneNumber": phone_number,
            "website": website
        }

def save_restaurant_data_to_tsv(data, output_file) -> None:
    """
    Save the extracted restaurant data to a TSV file.
    
    param data: list of dictionaries with the extracted data
    
    param output_file: string with the path to the output TSV file
    """
    # Define the TSV headers
    tsv_headers = [
        "restaurantName", "address", "city", "postalCode", "country",
        "priceRange", "cuisineType", "description", "facilitiesServices",
        "creditCards", "phoneNumber", "website"
    ]
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile: # open the TSV file
        writer = csv.DictWriter(tsvfile, fieldnames=tsv_headers, delimiter='\t') # create a CSV DictWriter object with tab delimiter
        writer.writeheader() # write the header row
        for row in data: # iterate over the data
            writer.writerow(row) # write each row to the TSV file

def forming_a_tsv():
    main_directory = 'michelin_html_batches' # Directory containing the sub-directories which then contain HTML files
    tsv_output_file = 'Files/michelin_restaurant_data.tsv' # Output TSV file to save the extracted data

    # Initialize list to hold all restaurant data
    all_restaurant_data = [] # list to store the extracted data

    # Traverse each batch directory and extract data from HTML files
    for batch_dir in os.listdir(main_directory): # iterate over the directories in the base output directory
        batch_path = os.path.join(main_directory, batch_dir) # create the path to the batch directory
        if os.path.isdir(batch_path): # check if the path is a directory
            for html_file in os.listdir(batch_path): # iterate over the files in the batch directory
                html_file_path = os.path.join(batch_path, html_file) # create the path to the HTML file
                if html_file_path.endswith('.html'): # check if the file is an HTML file
                    # print(f"Processing file: {html_file_path}") # for debugging purposes
                    restaurant_data = extract_restaurant_data(html_file_path) # extract data from the HTML file
                    all_restaurant_data.append(restaurant_data) # append the extracted data to the list

    save_restaurant_data_to_tsv(all_restaurant_data, tsv_output_file) # save the extracted data to a TSV file
    print(f"Data saved to {tsv_output_file}") 
