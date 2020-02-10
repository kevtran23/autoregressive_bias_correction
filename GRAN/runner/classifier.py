import torch 
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
import dgl
import dgl.function as fn
import networkx as nx
import pickle
from random import shuffle
import matplotlib as plt 

# Sends a message of node feature h.
msg = fn.copy_src(src='h', out='m')

def reduce(nodes):
    """Take an average over all neighbor node features hu and use it to
    overwrite the original node feature."""
    accum = torch.mean(nodes.mailbox['m'], 1)
    return {'h': accum}

class NodeApplyModule(nn.Module):
    """Update the node feature hv with ReLU(Whv+b)."""
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        h = self.activation(h)
        return {'h' : h}

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        # Initialize the node features with h.
        g.ndata['h'] = feature
        g.update_all(msg, reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()

        self.layers = nn.ModuleList([
            GCN(in_dim, hidden_dim, F.relu),
            GCN(hidden_dim, hidden_dim, F.relu) ])
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        # For undirected graphs, in_degree is the same as
        # out_degree.
        h = g.in_degrees().view(-1, 1).float()
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        hg = dgl.mean_nodes(g, 'h')
        return self.classify(hg)

# save a list of graphs
def save_graph_list(G_list, fname):
  print(fname)
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)

def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list

def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  # adj_list = G.adjacency_list()
  # for id,adj in enumerate(adj_list):
  #     id_min = min(adj)
  #     if id<id_min and id>=1:
  #     # if id<id_min and id>=4:
  #         break
  # node_list = list(range(id)) # only include node prior than node "id"

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list)
  G = max(nx.connected_component_subgraphs(G), key=len)
  return G

class GraphDataSet(Dataset):
    def __init__(self, train_samples_path, generative_samples_path):
        """
        path: path to the file containing pickle dump of a list of graphs 
        """
        self.train_samples_graph_list = load_graph_list(train_samples_path)
        self.generative_samples_graph_list = load_graph_list(generative_samples_path)
        self.dgl_graph_list = []
        
        for i in range(len(self.train_samples_graph_list)):
            g = dgl.DGLGraph()
            g.from_networkx(self.train_samples_graph_list[i])
            self.dgl_graph_list.append(g)
        
        for i in range(len(self.generative_samples_graph_list)):
            g = dgl.DGLGraph()
            g.from_networkx(self.generative_samples_graph_list[i])
            self.dgl_graph_list.append(g)

        self.labels = [0]*len(self.train_samples_graph_list) + [1]*len(self.generative_samples_graph_list)

    def __getitem__(self, index):
        """Take the index of item and returns the appropriate graph and its label"""
        return self.dgl_graph_list[index], self.labels[index]

    def __len__(self):
        return len(self.dgl_graph_list)

def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

train_samples_path = 'data/save_split/DD_train.p'
train_generative_samples_path = 'data/save_split/DD_generative_train.p'
test_samples_path = 'data/save_split/DD_test.p'
test_generative_samples_path = 'data/save_split/DD_generative_test.p'
train_set = GraphDataSet(train_samples_path, train_generative_samples_path)
test_set = GraphDataSet(test_samples_path, test_generative_samples_path)
train_loader = DataLoader(train_set, batch_size=8, shuffle=True,
                            collate_fn=collate)

model = Classifier(1, 256, 2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
model.train()

epoch_losses = []
for epoch in range(300):
    epoch_loss = 0
    for iter, (bg, label) in enumerate(train_loader):
        prediction = model(bg)
        loss = loss_func(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
    epoch_loss /= (iter + 1)
    print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
    epoch_losses.append(epoch_loss)
    torch.save(model.state_dict(),'snapshot_model/classifier.pth')

# train_path = 'data/save_split/DD_train.p'
# dev_path = 'data/save_split/DD_dev.p'
# all_graphs = load_graph_list(dev_path)
# print(list(all_graphs[0].edges), 'nodes 1')
# print(list(all_graphs[200].edges), 'nodes 2')
# print(list(all_graphs[400].edges), 'nodes 3')
# print(list(all_graphs[600].edges), 'nodes 4')
# print(list(all_graphs[800].edges), 'nodes 5')

# dev_path = 'data/save_split/DD_generative_train.p'
# all_graphs = load_graph_list(dev_path)
# save_graph_list(all_graphs[:734],'data/save_split/DD_generative_train.p')
# save_graph_list(all_graphs[734:],'data/save_split/sample_batch_1.p')
# train_graphs = load_graph_list('data/save_split/DD_generative_train.p')
# test_graphs = load_graph_list('data/save_split/sample_batch_1.p')
# print(len(train_graphs), 'length of train graphs')
# print(len(test_graphs), 'length of test graphs')

# test_path = 'data/save_split/DD_test.p'
# train_graphs = load_graph_list(train_path) 
# dev_graphs = load_graph_list(dev_path)
# test_graphs = load_graph_list(test_path)
# print(len(train_graphs), 'train len')
# print(len(dev_graphs), 'dev len')
# print(len(test_graphs), 'test len')

# path1 = 'data/save_split/sample_batch_1.p'
# path2 = 'data/save_split/sample_batch_2.p'
# path3 = 'data/save_split/sample_batch_3.p'
# path4 = 'data/save_split/sample_batch_4.p'
# path5 = 'data/save_split/sample_batch_5.p'
# graph1 = load_graph_list(path1) 
# graph2 = load_graph_list(path2) 
# graph3 = load_graph_list(path3) 
# graph4 = load_graph_list(path4) 
# graph5 = load_graph_list(path5) 
# all_graphs = graph1 + graph2 + graph3 + graph4 + graph5
# print(len(all_graphs), 'total num graphs')
# print(list(all_graphs[0].edges), 'nodes 1')
# print(list(all_graphs[1].edges), 'nodes 2')
# print(list(all_graphs[2].edges), 'nodes 3')
# print(list(all_graphs[3].edges), 'nodes 4')
# print(list(all_graphs[4].edges), 'nodes 5')
# save_graph_list(graphs_gen, 'data/save_split/DD_dev.p')
# dev_graphs = load_graph_list('data/save_split/DD_dev.p')
# # print(len(dev_graphs), 'total num graphs loaded')

# epoch_losses = []
# for epoch in range(300):
#     epoch_loss = 0
#     for iter, (bg, label) in enumerate(train_loader):
#         print(bg)
#         prediction = model(bg)
#         loss = loss_func(prediction, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.detach().item()
#     epoch_loss /= (iter + 1)
#     print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
#     epoch_losses.append(epoch_loss)
#     torch.save(model.state_dict(),'snapshot_model/classifier.pth')
#     print('saved')

# plt.title('cross entropy averaged over minibatches')
# plt.plot(epoch_losses)
# plt.show()
# model = Classifier(1, 512, 2)
# model.load_state_dict = torch.load('snapshot_model/classifier.pth')
# model.eval()
# # Convert a list of tuples to two lists
# test_X, test_Y = map(list, zip(*test_set))
# test_bg = dgl.batch(test_X)
# test_Y = torch.tensor(test_Y).float().view(-1, 1)
# probs_Y = torch.softmax(model(test_bg), 1)
# sampled_Y = torch.multinomial(probs_Y, 1)
# argmax_Y = torch.max(probs_Y, 1)[1].view(-1, 1)
# print('Accuracy of sampled predictions on the test set: {:.4f}%'.format(
#     (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100))
# print('Accuracy of argmax predictions on the test set: {:4f}%'.format(
#     (test_Y == argmax_Y.float()).sum().item() / len(test_Y) * 100)

# #trainset, testset = random_split(dataset, [len(dataset)-100, 100])
# data_loader = DataLoader(dataset, batch_size=1, shuffle=True,
#                          collate_fn=collate)

# model = Classifier(1, 256, 2)
# loss_func = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)
# model.train()

# epoch_losses = []
# for epoch in range(80):
#     epoch_loss = 0
#     for iter, (bg, label) in enumerate(data_loader):
#         prediction = model(bg)
#         loss = loss_func(prediction, label)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.detach().item()
#     epoch_loss /= (iter + 1)
#     print('Epoch {}, loss {:.4f}'.format(epoch, epoch_loss))
#     epoch_losses.append(epoch_loss)