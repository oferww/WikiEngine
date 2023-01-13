# WikiEngine
This repository contains the code for a search engine for the Wikipedia corpus, utilizing an inverted index for efficient querying.

## Files
inverted_index_gcp.py - includes the inverted index class and helper functions for building the index.
indexes_builder_run_in_cluster.ipynb - Jupyter notebook for building the inverted indexes using the inverted_index_gcp.py file and running in a cluster.
search_frontend.py - runs the search engine frontend on an instance using the inverted indexes built and stored in a Google Cloud Platform (GCP) bucket.

## Requirements
Python 3.x
Google Cloud SDK
gsutil

## Usage
Clone the repository.
Run the indexes_builder_run_in_cluster.ipynb to build the inverted indexes.
Run the search_frontend.py to start the search engine on an instance.
Use the provided UI to query the Wikipedia corpus.

## Note
Please note that the code is written using gcp specific libraries, so you may have to replace them with other cloud providers or local storage libraries.
