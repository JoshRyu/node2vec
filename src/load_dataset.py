from torch_geometric.datasets import AmazonProducts
from torch_geometric.utils.convert import to_networkx

dataset = AmazonProducts(root='../dataset')
data = dataset[0]
print(data)

G = to_networkx(data)
node_labels = data.y[list(G.nodes)].numpy()
print(node_labels)
node_class = {k: v for v, k in enumerate(node_labels)}
print(node_class)