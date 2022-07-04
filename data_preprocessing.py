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
import glob
import pickle

# Select number of threads to use
os.environ["OMP_NUM_THREADS"] = "2" # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = "2" # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = "2" # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = "2" # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = "2" # export NUMEXPR_NUM_THREADS=1

PID = os.getpid()
PGID = os.getpgid(PID)
print(f"PID: {PID}, PGID: {PGID}")

data_path = "./data/original/ud/"
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
    tokens_sentence.append(token.form)

    # Add edges from parent to current node and vice versa
    if( int(token.id) != 0):
        edges_start.append(int(token.head))
        edges_end.append(int(token.id))
        edges_start.append(int(token.id))
        edges_end.append(int(token.head))

    for subtree in tree.__iter__():
        DepTreeToPytorchGeom(subtree)

def createNetworkxEdgeAttributes(edge_attributes, edge_index, oh_labels, node_idx_list):
  edge_attrs_networkx = {}
  num_edges = edge_index[0].size()[0] - 1
  edge_attributes = oh_encoder_dependencies.inverse_transform(np.array(edge_attributes))

  for idx, edge_attr in enumerate(edge_attributes):
    if idx <= num_edges:
      # Get start node of edge
      start = edge_index[0][idx].item()
      start = node_idx_list[start]
      # Get end node of edge
      end = edge_index[1][idx].item()
      end = node_idx_list[end]
      # Get dependency tag
      label = edge_attributes[idx][-1]
      # Add label to list
      edge_attrs_networkx[(start,end)] = {"deprel": label}
  return edge_attrs_networkx

def CreateNetworkxNodeAttributes(node_attributes, node_idx_list):
  node_attrs_networkx = {}
  node_attributes = oh_encoder_dependencies.inverse_transform(np.array(node_attributes))

  for idx, node_attr in enumerate(node_attributes):
    # Get node index from list
    node_idx = node_idx_list[idx]
    # Add label to list
    node_attrs_networkx[node_idx] = node_attr[0]

  return node_attrs_networkx

def SavePyGeomGraphImage(data, filename):

    sentenceIdx = idx + 1

    graph = tg_utils.to_networkx(data)
    # Create depth-first tree from graph
    graph = nx.dfs_tree(graph, source=0)
    #list(nx.dfs_preorder_nodes(G, source=0))
    idx_order = list(graph.nodes)
    # Create networkx node labels with dependency relations
    node_attrs_networkx = CreateNetworkxNodeAttributes(data.x, idx_order)
    nx.set_node_attributes(graph, node_attrs_networkx, name="deprel")
    # Create networkx edge attributes with dependency relations
    # edge_attrs_networkx = createNetworkxEdgeAttributes(data.edge_attr, data.edge_index, dependency_tags, idx_order)
    #nx.set_edge_attributes(graph, edge_attrs_networkx)

    nx.nx_agraph.write_dot(graph,"./images/ud_graphs/graph"+str(sentenceIdx)+".dot")
    layout = graphviz_layout(graph, prog="dot")
    nx.draw(graph, layout, arrows=False, node_size=800, labels=node_attrs_networkx, with_labels=True)
    nx.draw_networkx_edges(graph, pos = layout)
    plt.savefig("./images/ud_graphs/"+ filename + str(sentenceIdx) +".png")
    plt.clf()
    # Remove dot file
    os.remove("./images/ud_graphs/graph"+str(sentenceIdx)+".dot")

# Convert dependency tags to one-hot labels
oh_encoder_dependencies = preprocessing.OneHotEncoder()
oh_encoder_dependencies.fit(np.array(dependency_tags).reshape(-1,1))

for ud_file in glob.iglob(data_path + '**/*.conllu', recursive=True):

  ud_file = os.path.abspath(ud_file)
  filename = os.path.basename(ud_file)
  print(filename)
  file = pyconll.load_from_file(ud_file)

  for idx, sentence in enumerate(file):
    # Edges for one graph
    edges_start = []
    edges_end = []
    # Dependency tags for a sentence (edge attribute). Add root node by default.
    dependency_tags_sentence = []
    dependency_tags_sentence.append("-")
    # Tokens in a sentence (node attribute)
    tokens_sentence = []
    tokens_sentence.append("root")

    raw_sentences.append(sentence.text)
    dep_tree = sentence.to_tree()
    DepTreeToPytorchGeom(dep_tree)

    # One-Hot encode dependency tags in sentence
    oh_dependency_tags = oh_encoder_dependencies.transform(np.array(dependency_tags_sentence).reshape(-1,1)).toarray()

    # Create Pytorch data object
    edge_index = torch.tensor([edges_start,edges_end], dtype=torch.long)

    # Add edge attributes: dependency tags
    # Duplicate dependency tags to create edge attributes list (undirected edges)
    # edge_attrs = [ i for i in oh_dependency_tags for r in range(2) ]

    # Add node attributes: dependency tags
    x = torch.tensor(oh_dependency_tags, dtype=torch.float)
    # edge_attrs = torch.tensor(edge_attrs, dtype=torch.float)

    data = Data(edge_index=edge_index, x=x)
    syntax_graphs.append(data)

    if(idx<=4):
      # Save graph image
      filename = filename.split(".")[0]
      SavePyGeomGraphImage(data, filename)

  # Save list of Pytorch geometric data objects
  filename = ud_file.split(".")[0] + ".syntree"
  filename = filename.replace("original/","")

  dirname = os.path.dirname(filename)
  if not os.path.exists(dirname):
      os.makedirs(dirname)

  with open(filename, 'wb') as handle:
      pickle.dump(syntax_graphs, handle)

  # Save raw corpus text
  filename = filename.split(".")[0] + ".txt"
  with open(filename, 'w') as output:
      output.write("\n".join(raw_sentences))
