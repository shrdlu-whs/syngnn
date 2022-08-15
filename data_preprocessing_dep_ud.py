####
# Data preprocessing for dependency grammar treebank data
# Processes Universal Dependencies *.conllu files
####

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

data_path = "./data/original/ud/"
# BERT tokenizer to use:
tokenizer_name = 'bert-base-uncased'
# Set of syntactic dependency tags
dependency_tags = ["-","root","punct","dep","nsubj","nsubj:pass","nsubj:outer","obj","iobj","csubj","csubj:pass","csubj:outer","ccomp","xcomp","nummod","appos","nmod","nmod:npmod","nmod:tmod","nmod:poss","acl","acl:relcl","amod","det","det:predet","case","obl","obl:npmod","obl:tmod","advcl","advmod","compound","compound:prt","fixed","flat","flat:foreign","goeswith","vocative","discourse","expl","aux","aux:pass","cop","mark","conj","cc","cc:preconj","parataxis","list","dislocated","orphan","reparandum", "obl:agent"]
# Raw text sentences
raw_sentences = []
# List of Pytorch Geometric syntax graphs
syntax_graphs = []

device =  torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained(tokenizer_name)


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

    #print(f"{token.head}->{token.id}, {token.upos}, {token.form}, {token.deprel}")

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
  #print(edge_attrs_networkx)
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
    #print(node_attrs_networkx)
    #nx.set_node_attributes(graph, node_attrs_networkx)
    # Create networkx edge attributes with dependency relations
    edge_attrs_networkx = createNetworkxEdgeAttributes(data.edge_attr, data.edge_index, dependency_tags, idx_order)
    #nx.set_edge_attributes(graph, edge_attrs_networkx)

    dirname = os.path.dirname("./images/ud_graphs/")
    if not os.path.exists(dirname):
      os.makedirs(dirname)
    filename_dot = filename + str(sentenceIdx) + f"-{tokenizer_name}.dot"
    filename_png = filename + str(sentenceIdx) + f"-{tokenizer_name}.png"
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

for ud_file in glob.iglob(data_path + '**/*.conllu', recursive=True):
  raw_sentences = []
  ud_file = os.path.abspath(ud_file)
  filename = os.path.basename(ud_file)
  print(filename)
  file = pyconll.load_from_file(ud_file)

  for sentence_idx, sentence in enumerate(file):
    # Edges for one graph
    edges_start = []
    edges_end = []
    # Dependency tags for a sentence (edge attribute). Add root node by default.
    dependency_tags_sentence = []
    #dependency_tags_sentence.append("-")
    # Tokens in a sentence (node attribute)
    words_graph = []
    words_graph.append("root")
    raw_sentence = sentence.text
    raw_sentences.append(raw_sentence)
    # Map of conll index to pytorch index
    conll_pytorch_idx_map = [0]
    dep_tree = sentence.to_tree()
    DepTreeToPytorchGeom(dep_tree)
    conll_pytorch_idx_dict = dict(enumerate(conll_pytorch_idx_map, 0))
    conll_pytorch_idx_dict = {v: k for k, v in conll_pytorch_idx_dict.items()}
    #print(conll_pytorch_idx_dict)

    #print("Before mapping of conllu to pytorch indizes:")
    #print(edges_start)
    #print(edges_end)

    for idx, start_node in enumerate(edges_start.copy()):
      edges_start[idx] = conll_pytorch_idx_dict[start_node]
    for idx, end_node in enumerate(edges_end.copy()):
      edges_end[idx] = conll_pytorch_idx_dict[end_node]
    
    #print(f"Index dict{conll_pytorch_idx_dict}")
    #print("After mapping of conllu to pytorch indizes:")
    #print(edges_start)
    #print(edges_end)
 

    # One-Hot encode dependency tags in sentence
    oh_dependency_tags = oh_encoder_dependencies.transform(np.array(dependency_tags_sentence).reshape(-1,1)).toarray()

    min_end = find_min(edges_start)
    max_end = find_max(edges_start)
    node_indices = range(min_end, max_end)

    # Create subtokens
    edges_start_tokenized = edges_start.copy()
    #edges_start_tokenized.extend(edges_start)
    edges_end_tokenized = edges_end.copy()
    
    #edges_end_tokenized.extend(edges_end)
    words_graph_tokenized = []
    words_graph_tokenized.extend(words_graph)
    ids_graph_tokenized = []
    words_sentence_tokenized = []

    insertion_count = 0

    # Tokenize sentence and add subword tokens to graph
    for node_idx in node_indices:
      word = words_graph[node_idx]
      tokens = tokenizer.tokenize(word)
      #print(tokens)
      for token_idx,token in enumerate(tokens):
         # Replace first token
        if(token_idx == 0):
          current_idx = node_idx +insertion_count
          words_graph_tokenized[current_idx] = token
          #insertion_count = insertion_count+1
        # Add connection to first subword token
        elif(token_idx == 1):
          # add subword token to end of token list
          words_graph_tokenized.append( token)
          subword_token_idx = len(words_graph_tokenized) -1

          # add subword token after token
          #current_idx = node_idx +insertion_count
          #words_graph_tokenized.insert(current_idx+1, token)
          #insertion_count = insertion_count+1
          # Increment values in edges_start and edges_end
          # print(f"before {edges_start_tokenized}")
          #edges_start_tokenized_temp = [z+1 if z > current_idx+1 else z for z in edges_start_tokenized]
          #edges_start_tokenized = edges_start_tokenized_temp
          #edges_end_tokenized_temp = [z+1 if z > current_idx+1  else z for z in edges_end_tokenized ] 
          #edges_end_tokenized = edges_end_tokenized_temp

          #print(f"after {edges_start_tokenized}")
          # Increment values in deprel
          #dependency_tags_sentence_temp = [z+1 if z > current_idx+1  else z for z in dependency_tags_sentence ] 
          #dependency_tags_sentence = dependency_tags_sentence_temp
          # add edge from last (sub)word token to current token and vv
          #subword_token_idx = current_idx+token_idx
          #edges_start_tokenized.append(current_idx)
          #edges_end_tokenized.append(subword_token_idx)
          #edges_start_tokenized.append(subword_token_idx)
          #edges_end_tokenized.append(current_idx)

          # add edge from last (sub)word token to current token and vv
          edges_start_tokenized.append(current_idx)
          edges_end_tokenized.append(subword_token_idx)
          edges_start_tokenized.append(subword_token_idx)
          edges_end_tokenized.append(current_idx)
        else:
            words_graph_tokenized.append( token)
            subword_token_idx = len(words_graph_tokenized) -1
            edges_start_tokenized.append(subword_token_idx-1)
            edges_end_tokenized.append(subword_token_idx)
            edges_start_tokenized.append(subword_token_idx)
            edges_end_tokenized.append(subword_token_idx-1)

    # raw_sentences[sentence_idx]
    # print(tokens_sentence)
    #print(edges_start)
    #print("After tokenization:")
    #print(edges_start_tokenized)
    #print(edges_end)
    #print(edges_end_tokenized)
    #print(words_graph_tokenized)

    # Convert graph tokens to ids
    ids_graph_tokenized = tokenizer.convert_tokens_to_ids(words_graph_tokenized)

    # Tokenize raw sentence
    # Add the special tokens.
    marked_text = "[CLS] " + raw_sentence+ " [SEP]"
    words_sentence_tokenized = tokenizer.tokenize(marked_text)
    ids_sentence_tokenized = tokenizer.convert_tokens_to_ids(words_sentence_tokenized)

    # Map sentence token ids to graph token ids
    sentence_graph_idx_map = {}
    ids_graph_tokenized_temp = ids_graph_tokenized.copy()
    #print(ids_sentence_tokenized)
    #print(ids_graph_tokenized_temp)

    for idx, token_id in enumerate(ids_sentence_tokenized):
      if token_id in ids_graph_tokenized_temp:
        token_idx_graph = ids_graph_tokenized_temp.index(token_id)
        ids_graph_tokenized_temp[token_idx_graph] = "token processed"
        sentence_graph_idx_map[idx] = token_idx_graph
    
    #print(sentence_graph_idx_map)
    tokens_graph = []
    # Get position of sentence tokens in graph 
    for idx, token_id in enumerate(ids_sentence_tokenized):
      if idx in sentence_graph_idx_map:
        graph_token_idx = ids_graph_tokenized[sentence_graph_idx_map[idx]]
        tokens_graph.append(tokenizer.convert_ids_to_tokens(graph_token_idx))
    
    #print("Original sentence:")
    #print(words_sentence_tokenized)
    #print("Reconstructed graph order:")
    #print(tokens_graph)

  

    # Create Pytorch data object
    edge_index = torch.tensor([edges_start_tokenized,edges_end_tokenized], dtype=torch.long)
    # Add edge attributes: dependency tags
    # Duplicate dependency tags to create edge attributes list (undirected edges)
    edge_attrs = [ i for i in oh_dependency_tags for r in range(2) ]
    edge_attrs = torch.tensor(np.array(edge_attrs), dtype=torch.float)

    # Add node attributes: sentence token ids
    ids_graph_tokenized_np = np.array(ids_graph_tokenized)
    #print(ids_graph_tokenized_np.shape)
    #print(ids_graph_tokenized_np[2])
    # Pad to embedding size
    ids_graph_tokenized_padded = np.zeros((ids_graph_tokenized_np.shape[0], 768))
    #print(np.zeros(ids_graph_tokenized_np.shape[0]))
    #ids_graph_tokenized_padded = [np.put(arr,[0],[ids_graph_tokenized_np[idx]]) for idx, arr in enumerate(ids_graph_tokenized_padded)]
    
    #ids_graph_tokenized_np = np.pad(ids_graph_tokenized_np, (0, 767), 'constant')

    #print(ids_graph_tokenized_padded)
    # Create x array of shape num_nodes, num_features
    #x = np.array_split(ids_graph_tokenized_padded, ids_graph_tokenized_padded.shape[1])
    #print(ids_graph_tokenized_np.shape)
    #print(np.array(edge_index).shape)
    #print(np.array(edge_attrs).shape)
    #print(edge_attrs)
    x = torch.tensor(ids_graph_tokenized_padded, dtype=torch.long)


    data = Data(x=x,edge_index=edge_index, edge_attr=edge_attrs)
    syntax_graphs.append([data, sentence_graph_idx_map])

    if(sentence_idx<=5):
      # Save graph image
      filename = filename.split(".")[0]
      #SavePyGeomGraphImage(data, filename)
      print(data)

  # Save raw corpus text
  filename_text = ud_file.split(".")[0] + f".txt"
  filename_text = filename_text.replace("original/","")

  dirname = os.path.dirname(filename_text)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  
  with open(filename_text, 'w') as output:
    output.write("\n".join(raw_sentences))

  # Save list of Pytorch geometric data objects
  filename_syntree = filename_text.split(".")[0] + f"-{tokenizer_name}.syntree"

  dirname = os.path.dirname(filename_syntree)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(filename_syntree, 'wb') as handle:
    pickle.dump(syntax_graphs, handle)


  
