import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

# Load Cora dataset
# You can download the Cora dataset from: https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz
# Extract the files and load the edges and node labels
edges_file = "../dataset/cora/cora.cites"
labels_file = "../dataset/cora/cora.content"

# Create a directed graph from the edges
G = nx.read_edgelist(edges_file, create_using=nx.DiGraph())

# Read node labels
node_labels = {}
with open(labels_file, 'r') as f:
    for line in f:
        data = line.strip().split('\t')
        node = data[0]
        label = data[-1]
        node_labels[node] = label

# Node2Vec parameters
p = 1.0  # Return parameter
q = 1.0  # In-out parameter
dimensions = 128  # Size of node embeddings
num_walks = 20  # Number of walks per node
walk_length = 5  # Length of each walk

# Generate walks
node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q)
model = node2vec.fit(window=10, min_count=1, workers=4)

# Get node embeddings
embeddings = {str(node): model.wv[str(node)] for node in G.nodes()}


# # Convert embeddings to DataFrame
# embedding_df = pd.DataFrame.from_dict(embeddings, orient='index')

# # Perform t-SNE for dimensionality reduction
# tsne = TSNE(n_components=2, random_state=42)
# embeddings_2d = tsne.fit_transform(embedding_df)


# label_encoder = LabelEncoder()
# numeric_labels = label_encoder.fit_transform(list(node_labels.values()))

# # Visualize the embeddings in 2D
# plt.figure(figsize=(10, 8))
# plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=numeric_labels, cmap='viridis')
# plt.title('Node2Vec Embeddings Visualization')
# plt.colorbar()
# plt.show()


# Prepare data for node classification
X = [embeddings[str(node)] for node in G.nodes()]
y = [node_labels[str(node)] for node in G.nodes()]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train an SVM classifier
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)