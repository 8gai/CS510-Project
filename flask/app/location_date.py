import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk import Tree
from nltk.stem import PorterStemmer
import numpy as np
from bs4 import BeautifulSoup
from urllib.request import urlopen
from datetime import datetime
from rank_bm25 import BM25Okapi

def extract_location(url):
  page = urlopen(url)
  html = page.read().decode("utf-8")
  soup = BeautifulSoup(html, "html.parser")
  stemmed_token = []
  score_list = [] #store score of each location
  res = []
  str = '' #article content
  for sentence in soup.find_all('p'):
          str += sentence.get_text()
  locations = get_continuous_chunks(str, "GPE") #look for location keywords
  
  return locations


def get_continuous_chunks(text, label):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text)
    clean_tokens = [w for w in tokens if not w in stop_words]
      
    chunked = ne_chunk(pos_tag(clean_tokens))
    prev = None
    continuous_chunk = []
    current_chunk = []

    for subtree in chunked:
        if type(subtree) == Tree and subtree.label() == label:
            current_chunk.append(" ".join([token for token, pos in subtree.leaves()]))
        if current_chunk: 
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append(named_entity)
                current_chunk = []
        else:
            continue

    return continuous_chunk