# Michelin Restaurants Scraper and Search Engine

## Algorithmic Methods of Data Mining (Sc.M. in Data Science). Homewok 3

### Team members
* Saif Ali 1936419
* Valeria Avino 1905974
* Luca Nudo 2027873
* Arman Salahshour 2168226

---

This repository contains the submission of the third homework for the course "Algorithmic Methods of Data Mining", for Group #8.

## Contents

### **Jupyter Notebook**
* __`main.ipynb`__  
   > The Jupyter notebook containing the solutions to all questions. The cells are already executed.

---

### **Python Scripts**
* __`crawler.py`__  
   > Contains functions related to scraping URLs of Michelin restaurants, fetching their HTML structure from the [Michelin Guide website](https://guide.michelin.com/en/it/restaurants/), and saving the HTML content to the local machine.

* __`myparser.py`__  
   > Contains functions for parsing relevant information from saved HTML files and preparing a `.csv` file with restaurant details.

* __`functions.py`__  
   > Contains utility functions used throughout the project, such as pre-processing and data manipulation, that are used in the main notebook.

---

### **Generated Files**
* __`vocabulary.csv`__  
   > Contains vocabulary words obtained by preprocessing the `description` column, with their associated `term_id`.

* __`inverted_index.json`__  
   > A dictionary where the entries are the `term_id`s of the vocabulary, and the keys are the lists of document IDs where each term appears.

* __`tf_idf_inverted_index.json`__  
   > A dictionary where each entry is a term, and the value is a list of tuples containing document IDs and their associated TF-IDF scores.

* __`restaurant_coordinates.csv`__  
   > Contains the geographical coordinates (latitude and longitude) of the Michelin restaurants scraped from the website.

* __`city_region_coordinates.csv`__  
   > Contains the geographical coordinates of regions only - relevant to the Michelin restaurant locations.

* __`urls.txt`__  
   > A text file with all the URLs of Michelin restaurant pages scraped from the website.

* __`test.txt`__  
   > A placeholder file created during the testing phase of the scraping pipeline.

---

### **HTML Visualizations**
* __`my_engine_results_map.html`__  
   > An interactive map visualization displaying the results of the custom-built search engine.

* __`michelin_restaurants_map.html`__  
   > An interactive map visualization showing all Michelin restaurant locations extracted from the dataset.

---
