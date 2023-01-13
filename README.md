# WikiEngine
This repository contains the code for a search engine for the Wikipedia corpus, utilizing an inverted index for efficient querying.

## Files
1. *inverted_index_gcp.py* - includes the inverted index class and helper functions for building and writing the index.
2. indexes_builder_run_in_cluster.ipynb - Jupyter notebook for building the inverted indexes using the inverted_index_gcp.py file and running in a cluster.
3. search_frontend.py - runs the search engine frontend on an instance using the inverted indexes built and stored in a Google Cloud Platform (GCP) bucket.

## Requirements
* Python 3.x
* Google Cloud SDK
* gsutil

## Usage
1. Clone the repository.
2. Run the indexes_builder_run_in_cluster.ipynb to build the inverted indexes.
3. Run the search_frontend.py to start the search engine on an instance.
4. Use the provided UI to query the Wikipedia corpus.

## Note

Please note that the code is written using gcp specific libraries, so you may have to replace them with other cloud providers or local storage libraries.
