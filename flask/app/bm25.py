from rank_bm25 import BM25Okapi
from textblob import Word
import json
import string
from tqdm import tqdm
import numpy as np
from app.location_date import extract_location
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag, ne_chunk
from nltk import Tree
from nltk.stem import PorterStemmer

from bs4 import BeautifulSoup
from urllib.request import urlopen
from datetime import datetime




def bm25_search(query):
    file_path = '../news_dataset.json'
    data = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in tqdm(lines):
            temp = []
            item = json.loads(line)
            category = item["category"]
            sentence = item["headline"].strip() + ". " + item["short_description"].strip()
            # stop_words = set(stopwords.words('english'))
            # tokens = word_tokenize(sentence)

            
            data.append([sentence, item])
    corpus = [doc[0].translate(str.maketrans('', '', string.punctuation)).replace('\n',"").lower() for doc in data]
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)

    corrected_query = []
    for word in query.split(" "):
        corrected_query.append(Word(word).correct())

    print("corrected: {}".format(corrected_query))


    tokenized_query = [word.lower() for word in corrected_query]
    print(tokenized_query)
    doc_scores = bm25.get_scores(tokenized_query)

    

    score_normalized = (doc_scores - doc_scores.min()) / (doc_scores.max() - doc_scores.min())

    num_articles = np.where(score_normalized>=0.7, 1, 0).sum()
    print("Relevant Articles: {}".format(num_articles))
    # print("Top {} Articles:".format(min(10, num_articles)))
    sort_idx = np.flip(np.argsort(doc_scores)[-min(10, num_articles):])
    result = []
    for idx in sort_idx:
        # print(score_normalized[idx], data[idx][1])
        link = data[idx][1]['link']
        # locations = extract_location(link)
        # data[idx][1]['location'] = locations
        result.append(data[idx][1])
    return result

