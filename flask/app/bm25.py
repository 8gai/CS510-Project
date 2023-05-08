from rank_bm25 import BM25Okapi
from textblob import Word
import json
import string
from tqdm import tqdm
import numpy as np



def bm25_search(query):
    file_path = '../news_dataset.json'
    data = []
    with open(file_path, 'r', encoding='utf-8') as fp:
        lines = fp.readlines()
        for line in tqdm(lines):
            item = json.loads(line)
            category = item["category"]
            sentence = item["headline"].strip() + ". " + item["short_description"].strip()
            data.append([sentence, item])
    corpus = [doc[0].translate(str.maketrans('', '', string.punctuation)).replace('\n',"").lower() for doc in data]
    tokenized_corpus = [doc.split(" ") for doc in corpus]

    bm25 = BM25Okapi(tokenized_corpus)
    tokenized_query = query.lower().split(" ")
    doc_scores = bm25.get_scores(tokenized_query)
    

    score_normalized = (doc_scores - doc_scores.min()) / (doc_scores.max() - doc_scores.min())

    num_articles = np.where(score_normalized>=0.7, 1, 0).sum()
    print("Relevant Articles: {}".format(num_articles))
    # print("Top {} Articles:".format(min(10, num_articles)))
    sort_idx = np.flip(np.argsort(doc_scores)[-num_articles:])
    result = []
    for idx in sort_idx:
        print(score_normalized[idx], data[idx][1])
        result.append(data[idx][1])
    return result

