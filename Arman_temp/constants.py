import os
import time
import requests 
import numpy as np
import pandas as pd
from tqdm import tqdm
from bs4 import BeautifulSoup

headers = {'User-Agent': 'Mozilla/5.0 (Linux; Android 5.1.1; SM-G928X Build/LMY47X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/47.0.2526.83 Mobile Safari/537.36'}
main_site = 'https://guide.michelin.com'
init_page = 'https://guide.michelin.com/en/it/restaurants'
pages_base_dir = '/home/ubuntu/TempRec/HW3/pages'