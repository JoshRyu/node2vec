#!/bin/bash

source C:/Users/Joshua/anaconda3/Scripts/activate py38

# DeepWalk
python C:/Repositories/node2vec/src/pagerank_node2vec.py citeseer deepwalk 5 cora_deepwalk

# Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py citeseer node2vec 5 cora_node2vec

# PageRank with Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py citeseer pagerank 5 cora_pagerank

