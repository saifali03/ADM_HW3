# Algorithmic Methods of Data Mining (Sc.M. in Data Science). Homewok 3

## Team members
* Saif Ali 1936419
* Valeria Avino 1905974
* Luca Nudo 2027873
* Arman Salahshour 2168226

The repository contains the submission of the third homework for the course "Algorithmic Methods of Data Mining", for the Group #8.
## Contents
* __`main.ipynb`__
    > The Jupyter notebook with the solutions to all the questions. The cells are already executed.
* __`crawler.py`__
    > Contatins functions relating to scraping urls of the Michelin restaurants, fetching their HTML structure from the [Michelin Guide website](https://guide.michelin.com/en/it/restaurants/), and saving the HTML structure to the local machine.
* __`myparser.py`__
  > Contatins functions relating to scraping relevant information from HTML saved into local machine and preparing a .csv file.
* __`vocabulary.csv`__
  > Contains vocabulary words obtained by preprocessing the description column, and their associated term_id
* __`inverted_index.json`__
  > Contains a dictionary where the entries are the term_id s of the vocabulary ad the keys are the list of document IDs where that term appears.
* __`tf_idf_inverted_index.json`__
  > Contains a new dictionary where each entry is a term, and the value is a list of tuples containing document IDs and TF-IDF scores.
* __`functions.py`__
* > Contains all the other functions used in the main notebook.
