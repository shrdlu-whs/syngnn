####
# Data preprocessing for constituency grammar treebank data
# Processes ICE-GB *.tre files
####


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
from transformers import BertTokenizer
from sklearn import preprocessing
import numpy as np
import os
import glob
import pickle
from utilities import find_min, find_max, inversePermutation

# Select number of threads to use
num_threads = "2"
os.environ["OMP_NUM_THREADS"] = num_threads # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = num_threads # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = num_threads # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = num_threads # export NUMEXPR_NUM_THREADS=1

# Limit no. of threads used by Pytorch
torch.set_num_threads = int(num_threads)

PID = os.getpid()
PGID = os.getpgid(PID)
print(f"PID: {PID}, PGID: {PGID}", flush=True)

data_path = "./data/original/ice-gb/"
# Set of syntactic dependency tags
dependency_tags = ["-","root","punct","dep","nsubj","nsubj:pass","nsubj:outer","obj","iobj","csubj","csubj:pass","csubj:outer","ccomp","xcomp","nummod","appos","nmod","nmod:npmod","nmod:tmod","nmod:poss","acl","acl:relcl","amod","det","det:predet","case","obl","obl:npmod","obl:tmod","advcl","advmod","compound","compound:prt","fixed","flat","flat:foreign","goeswith","vocative","discourse","expl","aux","aux:pass","cop","mark","conj","cc","cc:preconj","parataxis","list","dislocated","orphan","reparandum", "obl:agent"]
# Raw text sentences
raw_sentences = []
# List of Pytorch Geometric syntax graphs
syntax_graphs = []

device =  torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def DepTreeToPytorchGeom(tree):

    # Get data of root node
    token = tree.data
    # Add node dependency relation to list     
    dependency_tags_sentence.append(token.deprel)

    words_graph.append(token.form)
    conll_pytorch_idx_map.append(int(token.id))

    # Add edges from parent to current node and vice versa
    if( int(token.id) != 0):
        edges_start.append(int(token.head))
        edges_end.append(int(token.id))
        edges_start.append(int(token.id))
        edges_end.append(int(token.head))

    print(f"{token.head}->{token.id}, {token.upos}, {token.form}, {token.deprel}")

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
      #edge_attrs_networkx[(start,end)] = {"deprel": label}
      edge_attrs_networkx[(start,end)] = label
  print(edge_attrs_networkx)
  return edge_attrs_networkx

def CreateNetworkxNodeAttributes(node_attributes, node_idx_list):
  node_attrs_networkx = {}
  node_attributes = tokenizer.convert_ids_to_tokens(node_attributes)

  for idx, node_attr in enumerate(node_attributes):
    # Get node index from list
    node_idx = node_idx_list[idx]
    # Add label to list
    node_attrs_networkx[node_idx] = node_attr

  return node_attrs_networkx


def SavePyGeomGraphImage(data, filename):

    sentenceIdx = sentence_idx + 1

    graph = tg_utils.to_networkx(data)
    # Create depth-first tree from graph
    #graph = nx.dfs_tree(graph, source=0)
    idx_order = list(graph.nodes)
    # Create networkx node labels with tokens
    node_attrs_networkx = CreateNetworkxNodeAttributes(data.x, idx_order)
    print(node_attrs_networkx)
    #nx.set_node_attributes(graph, node_attrs_networkx)
    # Create networkx edge attributes with dependency relations
    edge_attrs_networkx = createNetworkxEdgeAttributes(data.edge_attr, data.edge_index, dependency_tags, idx_order)
    #nx.set_edge_attributes(graph, edge_attrs_networkx)

    dirname = os.path.dirname("./images/ud_graphs/")
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    filename_dot = filename + str(sentenceIdx) + ".dot"
    filename_png = filename + str(sentenceIdx) + ".png"
    filepath_dot = os.path.join(dirname, filename_dot)
    filepath_png =os.path.join(dirname, filename_png)

    nx.nx_agraph.write_dot(graph, filepath_dot)
    layout = graphviz_layout(graph, prog="dot")
    pos_attrs = {}
    for node, coords in layout.items():
      token_length = len(node_attrs_networkx[node])
      offset = 26 -token_length^2

      pos_attrs[node] = (coords[0] -offset, coords[1] - 2)

    nx.draw(graph, layout, arrows=False, with_labels=False)
    nx.draw_networkx_labels(graph, pos_attrs, labels=node_attrs_networkx, font_size = 9)
    nx.draw_networkx_edges(graph, pos = layout)

    nx.draw_networkx_edge_labels(graph, pos = layout, edge_labels=edge_attrs_networkx)
    plt.savefig(filepath_png)
    plt.clf()
    # Remove dot file
    os.remove(filepath_dot)

# Convert dependency tags to one-hot labels
oh_encoder_dependencies = preprocessing.OneHotEncoder()
oh_encoder_dependencies.fit(np.array(dependency_tags).reshape(-1,1))

class Node:
    def __init__(self, indented_line):
        self.children = []
        self.level = len(indented_line) - len(indented_line.lstrip())
        self.text = indented_line.strip()

    def add_children(self, nodes):
        childlevel = nodes[0].level
        while nodes:
            node = nodes.pop(0)
            if node.level == childlevel: # add node as a child
                self.children.append(node)
            elif node.level > childlevel: # add nodes as grandchildren of the last child
                nodes.insert(0,node)
                self.children[-1].add_children(nodes)
            elif node.level <= self.level: # this node is a sibling, no more children
                nodes.insert(0,node)
                return

    def as_dict(self):
        if len(self.children) > 1:
            return {self.text: [node.as_dict() for node in self.children]}
        elif len(self.children) == 1:
            return {self.text: self.children[0].as_dict()}
        else:
            return self.text

for filename in glob.iglob(data_path + '**/acad*.tre', recursive=True):
  with open(filename, encoding='cp1252') as ice_file:
    raw_sentences = []
    ice_filepath = os.path.abspath(filename)
    ice_filename = os.path.basename(ice_filepath)
    print(ice_filename)
    text = ice_file.read()
    root = Node('root')
    root.add_children([Node(line) for line in text.splitlines()[0:10] if line.strip()])
    d = root.as_dict()['root']
    print(d)  


  