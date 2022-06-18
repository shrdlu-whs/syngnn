# Load library for universal dependencies tree processing
import pyconll
# Load Pytorch Geometric
import torch
from torch_geometric.data import Data
import torch_geometric.utils as tg_utils
# Load networkx
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
# Load matplotlib.pyplot
import matplotlib.pyplot as plt
import pydot
import pygraphviz
from sklearn import preprocessing
import numpy as np
import os
import pickle


data_path = "./data/ud/ud_ewt/UD_English-EWT/"
ud_filename = "en_ewt-ud-test.conllu"
file = pyconll.load_from_file(data_path + ud_filename)

# Set of syntactic dependency tags
dependency_tags = ["-","root","punct","dep","nsubj","nsubj:pass","nsubj:outer","obj","iobj","csubj","csubj:pass","csubj:outer","ccomp","xcomp","nummod","appos","nmod","nmod:npmod","nmod:tmod","nmod:poss","acl","acl:relcl","amod","det","det:predet","case","obl","obl:npmod","obl:tmod","advcl","advmod","compound","compound:prt","fixed","flat","flat:foreign","goeswith","vocative","discourse","expl","aux","aux:pass","cop","mark","conj","cc","cc:preconj","parataxis","list","dislocated","orphan","reparandum"]
# Raw text sentences
raw_sentences = []
# List of Pytorch Geometric syntax graphs
syntax_graphs = []

def DepTreeToPytorchGeom(tree):

    # Get data of root node
    token = tree.data
    # Add node dependency relation to list     
    dependency_tags_sentence.append(token.deprel)

    # Add edges from parent to current node and vice versa
    if( int(token.id) != 0):
        edges_start.append(int(token.head))
        edges_end.append(int(token.id))
        edges_start.append(int(token.id))
        edges_end.append(int(token.head))

    for subtree in tree.__iter__():
        DepTreeToPytorchGeom(subtree)

def createNetworkxEdgeAttributes(edge_attributes, edge_index, oh_labels):
  edge_attrs_networkx = {}
  num_edges = edge_index[0].size()[0] - 1
  for idx, edge_attr in enumerate(edge_attributes):
    if idx <= num_edges:
      # Get start node of edge
      start = edge_index[0][idx].item()

      # Get end node of edge
      end = edge_index[1][idx].item()

      # Convert one-hot label
      label = oh_labels[edge_attr.argmax()]

      # Add label to list
      edge_attrs_networkx[(start,end)] = {"label": label}
  return edge_attrs_networkx

def CreateNetworkxNodeAttributes(node_attributes, oh_labels):
  node_attrs_networkx = {}
  num_edges = edge_index[0].size()[0] - 1
  node_attributes = oh_encoder.inverse_transform(np.array(node_attributes))
  for idx, node_attr in enumerate(node_attributes):

      # Add label to list
      node_attrs_networkx[idx] = node_attr[0]
  return node_attrs_networkx

def SavePyGeomGraphImage(data):

    # Create networkx node labels with dependency relations
    node_attrs_networkx = CreateNetworkxNodeAttributes(data.x, dependency_tags)
    graph = tg_utils.to_networkx(data)
    nx.set_node_attributes(graph, node_attrs_networkx, name="deprel")
    nx.nx_agraph.write_dot(graph,"./images/ud_graphs/graph"+str(idx)+".dot")
    layout = graphviz_layout(graph, prog="dot")
    nx.draw(graph, layout, arrows=False, node_size=800, labels=node_attrs_networkx, with_labels=True)
    plt.savefig("./images/ud_graphs/ud_graph"+str(idx)+".png")
    plt.clf()
    # Remove dot file
    os.remove("./images/ud_graphs/graph"+str(idx)+".dot")

# Convert dependency tags to one-hot labels
oh_encoder = preprocessing.OneHotEncoder()
oh_encoder.fit(np.array(dependency_tags).reshape(-1,1))

for idx, sentence in enumerate(file):
    # Edges for one graph
    edges_start = []
    edges_end = []
    # One-Hot Encoded dependency tags for graph
    dependency_tags_sentence = []
    dependency_tags_sentence.append("-")

    raw_sentences.append(sentence.text)
    print("Text: " + str(idx)+" " + sentence.text)
    dep_tree = sentence.to_tree()
    DepTreeToPytorchGeom(dep_tree)
    print(dependency_tags_sentence)

    # One-Hot encode dependency tags in sentence
    oh_dependency_tags = oh_encoder.transform(np.array(dependency_tags_sentence).reshape(-1,1)).toarray()

    # Create Pytorch data object
    edge_index = torch.tensor([edges_start,edges_end], dtype=torch.long)
    # Add Node features: dependency tags
    x = torch.tensor(oh_dependency_tags, dtype=torch.float)
    data = Data(edge_index=edge_index, x=x)
    syntax_graphs.append(data)
    # Save graph image
    SavePyGeomGraphImage(data)

    if(idx==30):
        break

# Save list of Pytorch geometric data objects
path = "./data/ud/ud_ewt/pyg/"
filename = ud_filename.split(".")[0] + ".pygdata"
with open(path + filename, 'wb') as handle:
    pickle.dump(syntax_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)