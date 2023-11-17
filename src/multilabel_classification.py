import pandas as pd
import networkx as nx
from node2vec import Node2Vec
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from datetime import datetime

# Load edges to create a graph
edges = pd.read_csv('../dataset/blogcatalog/edges.csv', header=None)
G = nx.from_pandas_edgelist(edges, source=0, target=1)

# Load nodes to obtain node IDs
nodes = pd.read_csv('../dataset/blogcatalog/nodes.csv', header=None)
node_ids = nodes[0].tolist()

# Load groups to obtain group IDs
groups = pd.read_csv('../dataset/blogcatalog/groups.csv', header=None)
group_ids = groups[0].tolist()

# Load group-edges to extract node-group memberships
group_edges = pd.read_csv('../dataset/blogcatalog/group-edges.csv', header=None)
node_groups = {node: [] for node in node_ids}
for _, row in group_edges.iterrows():
    node_groups[row[0]].append(row[1])

def format_time(start_time, end_time):
    time_difference = end_time - start_time
    total_seconds = time_difference.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)

def run_node2vec(p, q):
    # Generate Node2Vec embeddings
    node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, p=p, q=q)
    model = node2vec.fit(window=10, min_count=1, workers=8)

    # Prepare data for multi-label classification
    labels = [node_groups[node_id] for node_id in node_ids]
    mlb = MultiLabelBinarizer(classes=group_ids)
    encoded_labels = mlb.fit_transform(labels)

    # Get Node2Vec embeddings for nodes in the dataset
    embeddings = {str(node): model.wv[str(node)] for node in node_ids}

    # Convert embeddings to feature matrix
    X = [embeddings[str(node_id)] for node_id in node_ids]
    y = encoded_labels

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train multi-label classifier using SVM (you can try other classifiers as well)
    classifier = OneVsRestClassifier(SVC(kernel='linear'))
    classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = classifier.predict(X_test)

    # Evaluate F1-score (average='micro' considers all labels equally)
    f1 = f1_score(y_test, y_pred, average='micro')
    print(f"F1-score: {f1}")

    with open("../results/multilabel_classification.md", "a") as file:
        file.write(f'\nF1-score: {f1}')
        file.close()

start_time = datetime.now()
print('Start:', start_time)

for x in range(2):
  run_node2vec(0.001, 0.001)
  print("this is " + str(x) + " trial")

end_time = datetime.now()
print('End:', end_time)

print('Time taken:', format_time(start_time, end_time))