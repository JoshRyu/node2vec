import networkx as nx
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split, cross_val_score

from nodevectors.nodevectors import Node2Vec
from torch_geometric.datasets import Planetoid, AMiner
from torch_geometric.utils.convert import to_networkx

import argparse
import numpy as np
import textwrap
import pandas as pd
from datetime import datetime


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

def node_embedding(G, groups, p, q, mode, dimensions, num_walks, walk_length):
    node2vec_model = Node2Vec(n_components=dimensions, epochs=num_walks, walklen=walk_length)
    node2vec_model.fit(G, groups, p, q, mode)
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

def svm_prediction_for_periphery(embeddings, labels, group_indices):
    # Filter embeddings and labels for the specific group
    group_embeddings = [embeddings[index] for index in group_indices]
    group_labels = [labels[index] for index in group_indices]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(group_embeddings, group_labels, train_size=0.8, test_size=None, random_state=42)

    # SVM classifier
    svm_classifier = svm.SVC(kernel='rbf', C=8)
    svm_classifier.fit(X_train, y_train)
    y_pred = svm_classifier.predict(X_test)

    # Evaluation metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = svm_classifier.score(X_test, y_test)

    return macro_f1, micro_f1, accuracy

def load_dataset(name):
    name = name.casefold()
    data = None

    if (name == 'cora'):
        dataset = Planetoid(root='/tmp/Cora', name='Cora')

    if (name == 'citeseer'):
        dataset = Planetoid(root='/tmp/Citeseer', name='Citeseer')

    if (name == 'pubmed'):
        dataset = Planetoid(root='/tmp/PubMed', name='PubMed')


    data = dataset[0]
    
    return data

def load_testcase(method):
    method = method.casefold()
    test_plans = []

    if (method == 'deepwalk' or method == 'pagerank'):
        test_plans = [{'p': 1, 'q': 1}]

    if (method == 'node2vec'):
        test_plans = [{'p': .25, 'q': 4}]
        # {'p': .25, 'q': .5}, {'p': .25, 'q': .75}, {'p': .25, 'q': 1}, {'p': .25, 'q': 2}, {'p': .25, 'q': 4}, {'p': .5, 'q': .25}, {'p': .75, 'q': .25}, {'p': 1, 'q': .25}, {'p': 2, 'q': .25}, {'p': 4, 'q': .25}
    
    return test_plans

def load_mode(method):
    method = method.casefold()

    if (method in ['deepwalk', 'node2vec']):
        return 'base'
    
    else:
        return 'pagerank'

def find_maximum(groups):
    max_macro_f1_index = max(range(len(groups)), key=lambda i: groups[i]['macro_f1_score'])
    max_micro_f1_index = max(range(len(groups)), key=lambda i: groups[i]['micro_f1_score'])
    max_accuracy_index = max(range(len(groups)), key=lambda i: groups[i]['accuracy'])

    max_values = {
        'macro_f1_score': {
            'p': groups[max_macro_f1_index]['p'],
            'q': groups[max_macro_f1_index]['q'],
            'score': groups[max_macro_f1_index]['macro_f1_score']
        },
        'micro_f1_score': {
            'p': groups[max_micro_f1_index]['p'],
            'q': groups[max_micro_f1_index]['q'],  
            'score': groups[max_micro_f1_index]['micro_f1_score']
        },
        'accuracy': {
            'p': groups[max_accuracy_index]['p'],
            'q': groups[max_accuracy_index]['q'],
            'score': groups[max_accuracy_index]['accuracy']
        }
    }

    return max_values

def export_result_set(result_set, filepath, file_format='csv'):
    df = pd.DataFrame(result_set)
    if file_format.lower() == 'excel':
        df.to_excel(filepath + '.xlsx', index=False)
    elif file_format.lower() == 'csv':
        df.to_csv(filepath + '.csv', index=False)
    else:
        print("Unsupported file format. Please choose 'csv' or 'excel'.")

def format_time(start_time, end_time):
    time_difference = end_time - start_time
    total_seconds = time_difference.total_seconds()
    
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    milliseconds = int((time_difference.microseconds / 1000) % 1000)  # Extract milliseconds
    
    return '{:02d}:{:02d}:{:02d}.{:03d}'.format(hours, minutes, seconds, milliseconds)
    
def main(dataset, method, iteration, filename):
    data = load_dataset(dataset)

    G = to_networkx(data)
    node_labels_list = data.y.numpy()

    page_ranks = nx.pagerank(G, 0.8)

    # Divide nodes into groups based on PageRank levels
    group_a, group_b, group_c = divide_nodes_by_rank(page_ranks)
    
    # Calculate the starting index for the last 10%
    non_periphery = int(len(group_c) * 0.6)

    # Slice the list from the calculated index to the end
    group_d = group_c[non_periphery:]

    print(f'group A: {len(group_a)}')
    print(f'group B: {len(group_b)}')
    print(f'group C: {len(group_c)}')
    print(f'group D: {len(group_d)}')

    groups = {'A': group_a, 'B': group_b, 'C': group_c}

    # node_class = {k: v for v, k in enumerate(node_labels_list)}
    test_plans = load_testcase(method)
    iter_level = int(iteration)
    task_groups = []
    task_groups_periphery = []
    average_groups = []
    average_groups_periphery = []
    
    result_set = {
        'dataset': [],
        'technique': [],
        'dimensions': [],
        'num_of_walks': [],
        'walk_length': [],
        'p': [],
        'q': [],
        'macro_f1_score': [],
        'micro_f1_score': [],
        'accuracy': [], 
        'time_taken': [],
        'is_periphery': []
    }

    # Node2Vec parameters
    dimensions = 256  # Size of node embeddings 성능올릴거면 256
    num_walks = 10  # Number of walks per node
    walk_length = 10  # Length of each walk

    for index in range(len(test_plans)):

        tmp_macro_f1_score = []
        tmp_micro_f1_score = []
        tmp_accuracy = []

        tmp_peri_macro_f1_score = []
        tmp_peri_micro_f1_score = []
        tmp_peri_accuracy = []

        if (filename is not None):
            with open(f"../results/md/{filename}.md", "a") as file:
                file.write(f'#####################################\n')
                file.write(f'\n[## {method} experiments starts ##]\n')

        for iter in range(iter_level):

            start_time = datetime.now()
            # Node2Vec embedding
            embeddings = node_embedding(G, groups, test_plans[index]['p'], test_plans[index]['q'], load_mode(method), dimensions, num_walks, walk_length)

            macro_f1, micro_f1, accuracy = svm_prediction(embeddings, node_labels_list)
            macro_f1_periphery, micro_f1_periphery, accuracy_periphery = svm_prediction_for_periphery(embeddings, node_labels_list, group_d)

            print(f"'macro_f1': {macro_f1}, 'micro_f1': {micro_f1}, 'accuracy': {accuracy} ")
            print(f"\n'macro_f1_periphery': {macro_f1_periphery}, 'micro_f1_periphery': {micro_f1_periphery}, 'accuracy_periphery': {accuracy_periphery} ")

            end_time = datetime.now()
            
            tmp_macro_f1_score.append(macro_f1)
            tmp_micro_f1_score.append(micro_f1)
            tmp_accuracy.append(accuracy)

            tmp_peri_macro_f1_score.append(macro_f1_periphery)
            tmp_peri_micro_f1_score.append(micro_f1_periphery)
            tmp_peri_accuracy.append(accuracy_periphery)

            task_groups.append({'macro_f1_score': macro_f1, 'micro_f1_score': micro_f1, 'accuracy': accuracy, 'p': test_plans[index]['p'], 'q': test_plans[index]['q']})
            task_groups_periphery.append({'macro_f1_score': macro_f1_periphery, 'micro_f1_score': micro_f1_periphery, 'accuracy': accuracy_periphery, 'p': test_plans[index]['p'], 'q': test_plans[index]['q']})

            tmp_data = {
                'dataset': dataset,
                'technique': method,
                'dimensions': dimensions,
                'num_of_walks': num_walks,
                'walk_length': walk_length,
                'p': test_plans[index]['p'],
                'q': test_plans[index]['q'],
                'macro_f1_score': macro_f1,
                'micro_f1_score': micro_f1,
                'accuracy': accuracy,
                'time_taken': format_time(start_time, end_time),
                'is_periphery': False
            }

            tmp_data_periphery = {
                'dataset': dataset,
                'technique': method,
                'dimensions': dimensions,
                'num_of_walks': num_walks,
                'walk_length': walk_length,
                'p': test_plans[index]['p'],
                'q': test_plans[index]['q'],
                'macro_f1_score': macro_f1_periphery,
                'micro_f1_score': micro_f1_periphery,
                'accuracy': accuracy_periphery,
                'time_taken': format_time(start_time, end_time),
                'is_periphery': True
            }

            for key, value in tmp_data.items():
                result_set[key].append(value)

            for key, value in tmp_data_periphery.items():
                result_set[key].append(value)

            if (filename is not None):
                with open(f"../results/md/{filename}.md", "a") as file:
                    test_result = f"< Trial #{iter} > \nMacro_F1 Score: {macro_f1}\nMicro_F1 Score: {micro_f1}\nAccuracy Score: {accuracy}\n"
                    test_result += f"Macro_F1 Score for Perypheri: {macro_f1_periphery}\nMicro_F1 Score for Perypheri: {micro_f1_periphery}\nAccuracy Score for Perypheri: {accuracy_periphery}\n"
                    file.write(f'\n{test_result}')

        # Mean
        average_macro_score = np.mean(tmp_macro_f1_score)
        average_micro_score = np.mean(tmp_micro_f1_score)
        average_accuracy = np.mean(tmp_accuracy)

        # Mean
        average_peri_macro_score = np.mean(tmp_peri_macro_f1_score)
        average_peri_micro_score = np.mean(tmp_peri_micro_f1_score)
        average_peri_accuracy = np.mean(tmp_peri_accuracy)

        average_groups.append({'macro_f1_score': average_macro_score, 'micro_f1_score': average_micro_score, 'accuracy': average_accuracy, 'p': test_plans[index]['p'], 'q': test_plans[index]['q']})
        average_groups_periphery.append({'macro_f1_score': average_peri_macro_score, 'micro_f1_score': average_peri_micro_score, 'accuracy': average_peri_accuracy, 'p': test_plans[index]['p'], 'q': test_plans[index]['q']})

        # Variance
        var_macro_score = np.var(tmp_macro_f1_score)
        var_micro_score = np.var(tmp_micro_f1_score)
        var_accuracy = np.var(tmp_accuracy)

        # Standard Deviation
        std_macro_score = np.std(tmp_macro_f1_score)
        std_micro_score = np.std(tmp_micro_f1_score)
        std_accuracy = np.std(tmp_accuracy)

        if (filename is not None):
            with open(f"../results/md/{filename}.md", "a") as file:
                file.write(f"\n[ Mean Score for Task P: {test_plans[index]['p']}, Q: {test_plans[index]['q']} ] \nMacro_F1 Score: {average_macro_score}\nMicro_F1 Score: {average_micro_score}\nAccuracy Score: {average_accuracy}\n")
                file.write(f"\n[ Variance Score for Task P: {test_plans[index]['p']}, Q: {test_plans[index]['q']} ] \nMacro_F1 Score: {var_macro_score}\nMicro_F1 Score: {var_micro_score}\nAccuracy Score: {var_accuracy}\n")
                file.write(f"\n[ Standard Deviation Score for Task P: {test_plans[index]['p']}, Q: {test_plans[index]['q']} ] \nMacro_F1 Score: {std_macro_score}\nMicro_F1 Score: {std_micro_score}\nAccuracy Score: {std_accuracy}\n#####################################\n\n")
    
    task_group_max = find_maximum(task_groups)
    task_group_max_periphery = find_maximum(task_groups_periphery)
    avg_group_max = find_maximum(average_groups)
    avg_group_max_periphery = find_maximum(average_groups_periphery)

    if (filename is not None):
        with open(f"../results/md/{filename}.md", "a") as file:
            max_text = f"""\n[ Task that achieves maximum score ]
            Macro_F1 Score: {task_group_max['macro_f1_score']['score']}, p = {task_group_max['macro_f1_score']['p']}, q = {task_group_max['macro_f1_score']['q']}
            Micro_F1 Score: {task_group_max['micro_f1_score']['score']}, p = {task_group_max['micro_f1_score']['p']}, q = {task_group_max['micro_f1_score']['q']}
            Accuracy Score: {task_group_max['accuracy']['score']}, p = {task_group_max['accuracy']['p']}, q = {task_group_max['accuracy']['q']}
            Macro_F1 Score for Periphery: {task_group_max_periphery['macro_f1_score']['score']}, p = {task_group_max_periphery['macro_f1_score']['p']}, q = {task_group_max_periphery['macro_f1_score']['q']}
            Micro_F1 Score for Periphery: {task_group_max_periphery['micro_f1_score']['score']}, p = {task_group_max_periphery['micro_f1_score']['p']}, q = {task_group_max_periphery['micro_f1_score']['q']}
            Accuracy Score for Periphery: {task_group_max_periphery['accuracy']['score']}, p = {task_group_max_periphery['accuracy']['p']}, q = {task_group_max_periphery['accuracy']['q']}\n
            """

            avg_max_text = f"""\n[ Average that achieves maximum score ]
            Macro_F1 Score: {avg_group_max['macro_f1_score']['score']}, p = {avg_group_max['macro_f1_score']['p']}, q = {avg_group_max['macro_f1_score']['q']}
            Micro_F1 Score: {avg_group_max['micro_f1_score']['score']}, p = {avg_group_max['micro_f1_score']['p']}, q = {avg_group_max['micro_f1_score']['q']}
            Accuracy Score: {avg_group_max['accuracy']['score']}, p = {avg_group_max['accuracy']['p']}, q = {avg_group_max['accuracy']['q']}
            Macro_F1 Score for Periphery: {avg_group_max_periphery['macro_f1_score']['score']}, p = {avg_group_max_periphery['macro_f1_score']['p']}, q = {avg_group_max_periphery['macro_f1_score']['q']}
            Micro_F1 Score for Periphery: {avg_group_max_periphery['micro_f1_score']['score']}, p = {avg_group_max_periphery['micro_f1_score']['p']}, q = {avg_group_max_periphery['micro_f1_score']['q']}
            Accuracy Score for Periphery: {avg_group_max_periphery['accuracy']['score']}, p = {avg_group_max_periphery['accuracy']['p']}, q = {avg_group_max_periphery['accuracy']['q']}
            \n\n"""

            # Remove leading spaces from text strings
            max_text = '\n'.join([line.lstrip() for line in max_text.split('\n')])
            avg_max_text = '\n'.join([line.lstrip() for line in avg_max_text.split('\n')])

            file.write(f'[## {method} experiments ends ##]\n')
            file.write(max_text)
            file.write(avg_max_text)
    
        export_result_set(result_set, f"../results/csv/{filename}", 'csv')
        export_result_set(result_set, f"../results/excel/{filename}", 'excel')

    print(f"\n'task_group_max': {task_group_max} ")
    print(f"'task_group_max_periphery' : {task_group_max_periphery} ")
    print(f"'avg_group_max' : {avg_group_max} ")
    print(f"'avg_group_max_periphery' : {avg_group_max_periphery} \n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pagerank Node2Vec Script')
    parser.add_argument('dataset', choices=['cora', 'citeseer', 'pubmed'], help='Dataset name (Cora, CiteSeer or PubMed)')
    parser.add_argument('method', choices=['deepwalk', 'node2vec', 'pagerank'], help='Method name (DeepWalk, Node2Vec or PageRank)')
    parser.add_argument('iteration', type=int, help='Number of iterations')
    parser.add_argument('--filename', help='File for results in markdown format (without extension)', default=None)

    args = parser.parse_args()

    main(args.dataset, args.method, args.iteration, args.filename)