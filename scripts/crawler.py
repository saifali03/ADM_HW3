# import the relevant libraries
import requests
from bs4 import BeautifulSoup
import os
import re
import lxml
import csv
import random   
import time
from tqdm import tqdm
import asyncio
from tqdm.asyncio import tqdm
from aiohttp import ClientSession, ClientResponseError

def scraping_urls(url, pages):
    """
    This function scrapes the Michelin Guide website for the URLs of all the restaurants.
    
    param url: The base URL of the Michelin Guide website.
    param pages: The number of pages to scrape.
    
    return: A text file containing the URLs of all the restaurants.
    """
    # retrieved the header from inspect section of the website, then clicked on network tab and refreshed the page to get the header
    headers = {"Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
           "Accept-Encoding": "gzip, deflate, br, zstd",
           "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"}

    with open('Files/urls.txt', 'w') as file: # Open a text file to write the URLs
        for i in tqdm(range(1, pages + 1), desc="Scraping Pages"): # tqdm is used to show the progress bar
            try:
                time.sleep(random.uniform(1, 2)) # Random sleep time between 1 and 2 seconds
                current_page = url + str(i) # URL of the current page (first page for i=1)
                request = requests.get(current_page, headers=headers) # Send a GET request to the current page
                request.raise_for_status()  # Raise an error for any bad responses
                
                soup = BeautifulSoup(request.content, 'lxml') # Parse the HTML content of the current page
                for a in soup.select("div.flex-fill a"): # Select all the restaurant URLs on the current page
                    file.write("https://guide.michelin.com"+a.get('href') + '\n') # Write the URL to the text file
            
            except requests.exceptions.HTTPError as e: # Handle HTTP errors
                if request.status_code == 429:  # Too many requests
                    print("Too many requests made. Waiting for 10 seconds before retrying...")
                    for _ in range(3):  # Retry up to 3 times
                        time.sleep(10)  # Wait for 10 seconds before retrying
                        request = requests.get(current_page, headers=headers) # Retry the request
                        if request.status_code != 429: # If the request is successful
                            break # Exit the loop
                        else:
                            print(f"Too many requests on page {i}. Skipping...") # Skip the current page after 3 retries
                            continue  # Skip the current page after 3 retries
                else:
                    print(f"HTTP error on page {i}: {e}") # Print the error message
                    time.sleep(10) # Retry after 10 seconds if another HTTP error occurs
                    request = requests.get(current_page, headers=headers) # Retry the request
                    if request is None: # If the request is still unsuccessful
                        continue # Move to the next page
            
            except requests.exceptions.RequestException as e: # Handle request errors
                print(f"Request failed for page {i}: {e}") # Print the error message
                time.sleep(10)  # Retry after 10 seconds if another request error occurs
                request = requests.get(current_page, headers=headers) # Retry the request
                if request is None: # If the request is still unsuccessful
                    continue # Move to the next page
            
            except Exception as e:
                print(f"An error occurred on page {i}: {e}")
                time.sleep(10)  # Retry after 10 seconds if another error occurs
                request = requests.get(current_page, headers=headers) # Retry the request
                if request is None: # If the request is still unsuccessful
                    continue # Move to the next page


async def fetch_and_save(url, session, output_dir, file_name, retries=2):
    """
    Fetch HTML content from a URL and save it to a file.

    param session: aiohttp ClientSession object.
    param url: URL to fetch HTML content from.
    param output_dir: Directory to save the HTML file.
    param file_name: Name of the HTML file.
    param retries: Number of retries in case of errors. Default is 2.
    return: None
    """
    attempt = 0
    while attempt <= retries:
        try:
            # Try to fetch the HTML content from the URL
            async with session.get(url) as response:
                response.raise_for_status()  # Check for HTTP errors
                html_content = await response.text()  # Read the HTML content from the response

                # Ensure output directory exists
                os.makedirs(output_dir, exist_ok=True)
                
                # Write HTML content to file
                file_path = os.path.join(output_dir, f"{file_name}.html")
                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(html_content)

                print(f"Saved: {file_path}")  # Print the path of the saved HTML file
                return  # Exit the function if successful

        except ClientResponseError as e:
            if e.status == 429:
                print("Rate limit hit. Waiting before retrying...")
                await asyncio.sleep(10)  # Wait 10 seconds before retrying
            elif e.status >= 500:
                print(f"Server error for URL {url}: {e}")
                await asyncio.sleep(10)  # Retry after 10 seconds
            else:
                print(f"HTTP error for URL {url}: {e}")
                break  # Exit the loop for non-retryable errors

        except Exception as e:
            print(f"An error occurred: {e}")
            break  # Exit the loop for non-retryable errors
        
        attempt += 1  # Increment attempt counter

    print(f"Failed to fetch URL {url} after {retries} attempts")  # Print a failure message if all attempts fail



async def download_html_in_batches(urls, base_output_dir):
    """
    Download HTML files in batches from a list of URLs.
    
    param urls: List of URLs to download HTML files from.
    param base_output_dir: Base directory to save the HTML files.
    
    return: None"""
    BATCH_SIZE = 20  # Number of HTML files per folder
    async with ClientSession() as session: # Create an aiohttp ClientSession object
        with tqdm(total=len(urls), desc="Total Progress", unit="file") as pbar: # Create a progress bar
            for batch_num, i in enumerate(range(0, len(urls), BATCH_SIZE), start=1): # Loop over the URLs in batches
                batch_urls = urls[i:i + BATCH_SIZE] # Get the URLs for the current batch
            
                # Create directory for the current batch if it doesn't exist
                batch_dir = os.path.join(base_output_dir, f"batch_{batch_num}") # Path to the current batch directory
                os.makedirs(batch_dir, exist_ok=True) # Create the directory if it doesn't exist
            
                # Create tasks for downloading HTML files in this batch
                tasks = [] 
                for url in batch_urls:
                    restaurant_name = url.split('/')[-1]
                    task = fetch_and_save(url, session, batch_dir, restaurant_name)
                    tasks.append(task)
            
                # Wait for the batch of tasks to complete before moving to the next batch
                await asyncio.gather(*tasks) # Gather all the tasks for the current batch
                pbar.update(len(batch_urls))  # Update the progress bar with the number of files downloaded

def load_urls(file_path):
    """
    Load the URLs from a text file.
    
    param file_path: Path to the text file containing the URLs.
    
    return: List of URLs.
    """
    with open(file_path, 'r') as file: # Open the text file
        urls = [line.strip() for line in file if line.strip()] # Read the URLs from the file
    return urls # Return the list of URLs
