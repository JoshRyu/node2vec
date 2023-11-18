import networkx as nx
from node2vec import Node2Vec
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import f1_score
from scipy import io
from datetime import datetime

def format_time(start_time, end_time):
    time_difference = end_time - start_time
    total_seconds = time_difference.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    
    return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)


# Load the dataset
data = io.loadmat("../dataset/wikipedia/POS.mat")

# Assuming 'network' matrix represents the graph structure
network_matrix = data['network']

def run_node2vec(p,q):
    # Convert the sparse matrix to a NetworkX graph
    G = nx.from_scipy_sparse_matrix(network_matrix)

    # Generate node2vec embeddings
    node2vec = Node2Vec(G, dimensions=128, walk_length=80, num_walks=10, p=p, q=q)
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
    
    with open("../results/multilabel_classification.md", "a") as file:
        file.write(f'\nF1-score: {f1}')
        file.close()


start_time = datetime.now()
print('Start:', start_time)

for x in range(2):
    run_node2vec(1, 1)
    print("this is " + str(x) + "th trial")

end_time = datetime.now()
print('End:', end_time)

print('Time taken:', format_time(start_time, end_time))
