[2023-11-16 Experiment Report]

- Implement Node2Vec technique on blogcatalog dataset
- As node2vec algorithm is technically same as deepwalk when the return parameter and in-out parameter values are 1 each,
  deepwalk and node2vec algorithm are used for comparison of F1-score result.
- Under the identical condition (parameters), I reckon each experiment should take at least 100 times and find averages for it, yet each experiment takes at least 40 minutes so it needs to be find out the way to speedup this process.
  - Node2Vec library provides 'number of workers' variable but it seems not working properly.

[Deepwalk]
p=1, q=1
F1-score: 0.23709483793517408

[Node2Vec]
p=0.01, q=0.01
F1-score: 0.2435820895522388
