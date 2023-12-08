#!/bin/bash

source C:/Users/Joshua/anaconda3/Scripts/activate py38

# DeepWalk
python C:/Repositories/node2vec/src/pagerank_node2vec.py cora deepwalk 100 cora_deepwalk

# Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py cora node2vec 100 cora_node2vec

# PageRank with Node2Vec
python C:/Repositories/node2vec/src/pagerank_node2vec.py cora pagerank 100 cora_pagerank

