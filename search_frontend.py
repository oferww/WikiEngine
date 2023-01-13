from flask import Flask, request, jsonify


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False

import math
import itertools
from itertools import chain, islice, count, groupby
import time
import numpy as np
import re
import pandas as pd
import pickle
import nltk
from pathlib import Path
from google.cloud import storage
import sys
from collections import Counter, OrderedDict, defaultdict
import os
from operator import itemgetter
from contextlib import closing
nltk.download('stopwords')
from nltk.corpus import stopwords
from google.cloud import storage

client = storage.Client()
bucket_name = 'porjectirofererez111'
bucket = client.bucket(bucket_name)


body_blob = bucket.blob("postings_gcp/.body/body_inverted_index.pkl")
titles_blob = bucket.blob("postings_gcp/.titles/titles_inverted_index.pkl")
anchor_blob = bucket.blob("postings_gcp/.anchor/anchor_inverted_index.pkl")


with body_blob.open('rb') as f:
    body_inverted_index = pickle.load(f)

with titles_blob.open('rb') as f:
    titles_inverted_index = pickle.load(f)

with anchor_blob.open('rb') as f:
    anchor_inverted_index = pickle.load(f)

pv_blob = bucket.blob('pageviews-202108-user.pkl')
with pv_blob.open('rb') as f:
    wid2pv = pickle.loads(f.read())

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ['category', 'references', 'also', 'links', 'extenal', 'see', 'thumb']
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
all_stopwords = english_stopwords.union(corpus_stopwords)

TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
BLOCK_SIZE = 1999998


class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self):
        self._open_files = {}

    def read(self, locs, n_bytes, folder_str):
        b = []
        for f_name, offset in locs:
            if f_name not in self._open_files:
                blob = bucket.blob(f"postings_gcp/{f_name}")
                self._open_files[f_name] = blob.open("rb")
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def read_posting_list(inverted, w, folder_str):
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, folder_str)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            if doc_id == 0:
                break
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list

@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = query.lower()
    split = re.findall(r"[\w']+", query)
    querysplit = [x for x in split if x not in list(all_stopwords)]
    dic = {}
    for w in querysplit:
        try:
            lst = read_posting_list(titles_inverted_index, w, 'titles')
            for tup in lst:
                if tup[0] in dic:
                    dic[tup[0]] = dic[tup[0]] + 25
                else:
                    dic[tup[0]] = 25

            lst = read_posting_list(anchor_inverted_index, w, 'anchor')
            for tup in lst:
                if tup[0] in dic:
                    dic[tup[0]] = dic[tup[0]] + 50
                else:
                    dic[tup[0]] = 50

        except:
            continue

    sortd = sorted([(doc_id, np.round(score, 5)) for doc_id, score in dic.items()], key=lambda x: x[1], reverse=True)[:100]
    res = [(x[0], body_inverted_index.ids_titles_map[x[0]]) for x in sortd if x[0] in body_inverted_index.ids_titles_map.keys()]
    # END SOLUTION
    return jsonify(res[:100])


@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the
        staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = query.lower()
    split = re.findall(r"[\w']+", query)
    querysplit = [x for x in split if x not in list(all_stopwords)]
    dic = {}
    for w in querysplit:
        try:
            lst = read_posting_list(body_inverted_index, w, 'body')
            idf = np.log10(len(body_inverted_index.DL) / body_inverted_index.df[w])

        except:
            continue

        for tup in lst:
            if tup[0] in dic:
                dic[tup[0]] = dic[tup[0]] + (tup[1] * idf)
            else:
                dic[tup[0]] = tup[1] * idf

    for d, num in dic.items():
        dic[d] = dic[d] * (1/len(querysplit)) * (1/body_inverted_index.DL[d])

    sortd = sorted([(doc_id,np.round(score,5)) for doc_id, score in dic.items()], key = lambda x: x[1],reverse=True)[:100]
    res = [(x[0], body_inverted_index.ids_titles_map[x[0]]) for x in sortd]
    # END SOLUTION
    return jsonify(res)


@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title. DO NOT use stemming. DO
        USE the staff-provided tokenizer from Assignment 3 (GCP part) to do the
        tokenization and remove stopwords. For example, a document
        with a title that matches two distinct query words will be ranked before a
        document with a title that matches only one distinct query word,
        regardless of the number of times the term appeared in the title (or
        query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    query = query.lower()
    split = re.findall(r"[\w']+", query)
    querysplit = [x for x in split if x not in list(all_stopwords)]
    dic = {}
    for w in querysplit:
        try:
            lst = read_posting_list(titles_inverted_index, w, 'titles')
        except:
            continue

        for tup in lst:
            if tup[0] in dic:
                dic[tup[0]] = dic[tup[0]] + 1
            else:
                dic[tup[0]] = 1

    sortd = sorted([(doc_id, np.round(score, 5)) for doc_id, score in dic.items()], key=lambda x: x[1], reverse=True)
    res = [(x[0], body_inverted_index.ids_titles_map[x[0]]) for x in sortd]

    # END SOLUTION
    return jsonify(res)


@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE ANCHOR TEXT of articles, ordered in descending order of the
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page.
        DO NOT use stemming. DO USE the staff-provided tokenizer from Assignment
        3 (GCP part) to do the tokenization and remove stopwords. For example,
        a document with a anchor text that matches two distinct query words will
        be ranked before a document with anchor text that matches only one
        distinct query word, regardless of the number of times the term appeared
        in the anchor text (or query).

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    if len(query) == 0:
        return res
    # BEGIN SOLUTION
    query = query.lower()
    split = re.findall(r"[\w']+", query)
    querysplit = [x for x in split if x not in list(all_stopwords)]
    dic = {}
    for w in querysplit:
        try:
            lst = read_posting_list(anchor_inverted_index, w, 'anchor')
        except:
            continue

        for tup in lst:
            if tup[0] in dic:
                dic[tup[0]] = dic[tup[0]] + 1
            else:
                dic[tup[0]] = 1

    sortd = sorted([(doc_id, np.round(score, 5)) for doc_id, score in dic.items()], key=lambda x: x[1], reverse=True)
    res = [(x[0], body_inverted_index.ids_titles_map[x[0]]) for x in sortd if x[0] in body_inverted_index.ids_titles_map.keys()]
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank values for a list of provided wiki article IDs. 

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for wiki in wiki_ids:
        res.append(titles_inverted_index.pr_dic[wiki])
    # END SOLUTION
    return jsonify(res)


@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    for wiki in wiki_ids:
        res.append(wid2pv[wiki])
    # END SOLUTION
    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
