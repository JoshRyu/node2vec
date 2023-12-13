#!/bin/bash

source C:/Users/Joshua/anaconda3/Scripts/activate py38

# DeepWalk
python C:/Repositories/node2vec/src/pagerank_node2vec.py pubmed deepwalk 100 pubmed_deepwalk

# Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py pubmed node2vec 100 pubmed_node2vec

# PageRank with Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py pubmed pagerank 100 pubmed_pagerank

