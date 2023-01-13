# WikiEngine
This repository contains the code for running a search engine on the Wikipedia corpus, utilizing three inverted index for efficient querying, on a GCP project.

## Structure
### Files
1. **inverted_index_gcp.py** - includes the inverted index class and helper functions for building and writing the index.
2. **indexes_builder_run_in_cluster.ipynb** - Jupyter notebook for building three inverted indexes using the inverted_index_gcp.py, running in a GCP cluster.
3. **search_frontend.py** - runs the search engine frontend on a GCP instance using the three inverted indexes built and stored in a GCP bucket.
4. **startup_script_gcp.sh** - a shell script that sets up the Compute Engine instance.
5. **run_frontend_in_gcp.sh** - command-line instructions for deploying your search engine to GCP. 

## Requirements
* Python 3.x
* Google Cloud SDK
* gsutil

## Usage
1. Clone the repository.
2. Upload inverted_index_gcp.py and indexes_builder_run_in_cluster.ipynb to a GCP cluster.
3. Run the indexes_builder_run_in_cluster.ipynb to build the three inverted indexes.
4. Upload startup_script_gcp.sh and run_frontend_in_gcp.sh to a GCP shell.
5. Run the search_frontend.py with executing run_frontend_in_gcp.sh in a GCP shell.
6. Use the provided UI to query the Wikipedia corpus.

## Note

Please note that the code is written using GCP specific libraries, so you may have to replace them with other cloud providers or local storage libraries.
