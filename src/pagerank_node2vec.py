import networkx as nx
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score

from nodevectors.nodevectors import Node2Vec
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.convert import to_networkx

import textwrap

# Divide nodes into groups based on PageRank levels
def divide_nodes_by_rank(page_ranks):
    sorted_nodes = sorted(page_ranks.items(), key=lambda x: x[1], reverse=True)
    total_nodes = len(sorted_nodes)
    top_10_percent = int(total_nodes * 0.1)
    next_30_percent = int(total_nodes * 0.3)
    group_a = [node for node, _ in sorted_nodes[:top_10_percent]]
    group_b = [node for node, _ in sorted_nodes[top_10_percent:next_30_percent]]
    group_c = [node for node, _ in sorted_nodes[next_30_percent:int(total_nodes)+1]]
    return group_a, group_b, group_c

def node_embedding(G, groups, p, q):
    # Node2Vec parameters
    p = 1.0  # Return parameter
    q = 1.0  # In-out parameter
    dimensions = 256  # Size of node embeddings 성능올릴거면 256
    num_walks = 10  # Number of walks per node
    walk_length = 10  # Length of each walk

    node2vec_model = Node2Vec(n_components=dimensions, epochs=num_walks, walklen=walk_length)
    node2vec_model.fit(G, groups, p, q)
    embeddings = []
    for embedding in range(len(G)):
        embeddings.append(node2vec_model.predict(embedding))

    return embeddings

def svm_prediction(embeddings, labels):
    X_train, X_test, y_train, y_test = train_test_split(embeddings, labels, train_size=0.8, test_size=None, random_state=42)
    
    svm_classifier = svm.SVC(kernel = 'rbf', C=8)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)


    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = svm_classifier.score(X_test, y_test)

    return macro_f1, micro_f1, accuracy

dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

G = to_networkx(data)
node_labels_list = data.y.numpy()

page_ranks = nx.pagerank(G, 0.8)

# Divide nodes into groups based on PageRank levels
group_a, group_b, group_c = divide_nodes_by_rank(page_ranks)

groups = {'A': group_a, 'B': group_b, 'C': group_c}


# node_class = {k: v for v, k in enumerate(node_labels_list)}

test_plans = [{'p': .25, 'q': .25}, {'p': .25, 'q': .5}, {'p': .25, 'q': .75}, {'p': .25, 'q': 1}, {'p': .25, 'q': 2}, {'p': .25, 'q': 4}, {'p': .5, 'q': .25}, {'p': .75, 'q': .25}, {'p': 1, 'q': .25}, {'p': 2, 'q': .25}, {'p': 4, 'q': .25}]
iter_level = 2

for index in range(len(test_plans)):
    tmp_macro_f1_score = 0
    tmp_micro_f1_score = 0
    tmp_accuracy = 0

    with open("../results/pagerank2.md", "a") as file:
        file.write(f'#####################################')

    for iter in range(iter_level):
        # Node2Vec embedding
        embeddings = node_embedding(G, groups, test_plans[index]['p'], test_plans[index]['q'])
        macro_f1, micro_f1, accuracy = svm_prediction(embeddings, node_labels_list)
        
        tmp_macro_f1_score += macro_f1
        tmp_micro_f1_score += micro_f1
        tmp_accuracy += accuracy

        with open("../results/pagerank2.md", "a") as file:
            test_result = f"< Trial #{iter} > \nMacro_F1 Score: {macro_f1}\nMicro_F1 Score: {micro_f1}\nAccuracy Score: {accuracy}\n"
            file.write(f'\n{test_result}')


    with open("../results/pagerank2.md", "a") as file:
        average_macro_score = tmp_macro_f1_score / iter_level
        average_micro_score = tmp_micro_f1_score / iter_level
        average_accuracy = tmp_accuracy / iter_level

        file.write(f"\n[ Average Score for Task P: {test_plans[index]['p']}, Q: {test_plans[index]['q']} ] \nMacro_F1 Score: {average_macro_score}\nMicro_F1 Score: {average_micro_score}\nAccuracy Score: {average_accuracy}\n#####################################\n\n")
        file.close()