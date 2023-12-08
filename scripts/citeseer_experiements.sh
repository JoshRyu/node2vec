#!/bin/bash

source C:/Users/Joshua/anaconda3/Scripts/activate py38

# DeepWalk
python C:/Repositories/node2vec/src/pagerank_node2vec.py citeseer deepwalk 100 citeseer_deepwalk

# Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py citeseer node2vec 100 citeseer_node2vec

# PageRank with Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py citeseer pagerank 100 citeseer_pagerank

