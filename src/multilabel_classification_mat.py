import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from scipy import io

# Load the dataset
data = io.loadmat("../dataset/wikipedia/POS.mat")

# Assuming 'network' matrix represents the graph structure
network_matrix = data['network']

# Convert the sparse matrix to a NetworkX graph
G = nx.from_scipy_sparse_matrix(network_matrix)

# Generate node2vec embeddings
node2vec = Node2Vec(G, dimensions=64, walk_length=30, num_walks=200, workers=4)  # You can adjust parameters here
model = node2vec.fit(window=10, min_count=1, batch_words=4)

# Get node embeddings
node_embeddings = {node: model.wv[node] for node in G.nodes()}
X = [node_embeddings[i] for i in range(len(node_embeddings))]

# Assuming 'group' matrix represents the labels
labels_matrix = data['group']
y = [labels_matrix[i, 0] for i in range(len(node_embeddings))]

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train multi-label classifier using SVM
classifier = OneVsRestClassifier(SVC(kernel='linear'))
classifier.fit(x_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(x_test)

# Evaluate F1-score (average='micro' considers all labels equally)
f1 = f1_score(y_test, y_pred, average='micro')
print(f"F1-score: {f1}")
