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
import importlib
import nltk
#from nltk.tokenize import word_tokenize
#nltk.download('punkt')
import utilities_data_preprocessing as utils

# Reload utils library if changed
importlib.reload(utils)

# Select number of threads to use
num_threads = "4"
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
data_path_dev = "./data_sample/original/ud/"
data_path = "./data_sample/original/ud/"
# BERT tokenizer to use:
tokenizer_name = 'bert-base-cased'
# Set of syntactic universal dependency tags
dependency_tags = ["-","sub","root","punct","dep","nsubj","nsubj:pass","nsubj:outer","obj","iobj","csubj","csubj:pass","csubj:outer","ccomp","xcomp","nummod","appos","nmod","nmod:npmod","nmod:tmod","nmod:poss","acl","acl:relcl","amod","det","det:predet","case","obl","obl:npmod","obl:tmod","advcl","advmod","compound","compound:prt","fixed","flat","flat:foreign","goeswith","vocative","discourse","expl","aux","aux:pass","cop","mark","conj","cc","cc:preconj","parataxis","list","dislocated","orphan","reparandum", "obl:agent"]
# Universal dependencies Part of Speech tags
upos_tags = ["ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"]

device =  torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
print_graph = False
syntree_mode = "gold"
count_graph_sentence_discrepancy = 0

def dep_tree_to_pytorch_geom(tree):

    # Get data of root node
    token = tree.data
    # Add node dependency relation to list     
    dependency_tags_sentence.append(token.deprel)
    upos_tags_sentence.append(token.upos)

    words_graph.append(token.form)
    conll_pytorch_idx_map.append(int(token.id))

    # Add edges from parent to current node
    if( int(token.id) != 0):
        edges_start.append(int(token.head))
        edges_end.append(int(token.id))

    for subtree in tree.__iter__():
        dep_tree_to_pytorch_geom(subtree)

def create_networkx_edge_attributes(edge_attributes, edge_index, oh_labels, node_idx_list):
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

def create_networkx_node_attributes(node_attributes, node_idx_list):
  node_attrs_networkx = {}

  for idx, node_attr in enumerate(node_attributes):
    # Get node index from list
    node_idx = node_idx_list[idx]
    # Add label to list
    node_label = tokenizer.convert_ids_to_tokens([node_attr[0]])[0]
    node_attrs_networkx[node_idx] = node_label

  return node_attrs_networkx

# Save Pytorch Geometric Data object as png image
def save_pygeom_graph_image(data, filename):

    sentenceIdx = sentence_idx + 1

    graph = tg_utils.to_networkx(data)
    # Create depth-first tree from graph
    #graph = nx.dfs_tree(graph, source=0)
    idx_order = list(graph.nodes)
    # Create networkx node labels with tokens
    node_attrs_networkx = create_networkx_node_attributes(data.x, idx_order)
    #print(node_attrs_networkx)
    #nx.set_node_attributes(graph, node_attrs_networkx)
    # Create networkx edge attributes with dependency relations
    edge_attrs_networkx = create_networkx_edge_attributes(data.edge_attr, data.edge_index, dependency_tags, idx_order)
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
      #offset_1 = 26 -token_length^2
      #offset_2 = 2
      offset_1 = 0
      offset_2 = 0
      pos_attrs[node] = (coords[0] -offset_1, coords[1] - offset_2)

    nx.draw(graph, layout, arrows=False, with_labels=False, node_color='#d6f4fb', node_size=200)
    nx.draw_networkx_labels(graph, pos_attrs, labels=node_attrs_networkx, font_size = 9)
    nx.draw_networkx_edges(graph, pos = layout)

    nx.draw_networkx_edge_labels(graph, pos = layout, edge_labels=edge_attrs_networkx, font_size=8)
    plt.savefig(filepath_png)
    plt.clf()
    # Remove dot file
    os.remove(filepath_dot)


# Convert dependency tags to one-hot labels
oh_encoder_dependencies = preprocessing.OneHotEncoder()
oh_encoder_dependencies.fit(np.array(dependency_tags).reshape(-1,1))
unequal_length_count = 0
for ud_file in glob.iglob(data_path + '**/*.conllu', recursive=True):
  # Raw text sentences
  raw_sentences = []
  # Processed text sentences
  processed_sentences = []
  # Unresolved sentences indices
  unresolved_sentences = []
  # List of Pytorch Geometric syntax graphs
  syntax_graphs = []
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
    upos_tags_sentence = []
    #dependency_tags_sentence.append("-")
    # Tokens in a sentence (node attribute)
    words_graph = []
    words_graph.append("root")
    raw_sentence = sentence.text
    raw_sentences.append(raw_sentence)
    # Map of conll index to pytorch index
    conll_pytorch_idx_map = [0]
    dep_tree = sentence.to_tree()
    dep_tree_to_pytorch_geom(dep_tree)
    conll_pytorch_idx_dict = dict(enumerate(conll_pytorch_idx_map, 0))
    conll_pytorch_idx_dict = {v: k for k, v in conll_pytorch_idx_dict.items()}
    #print(conll_pytorch_idx_dict)

    #print("Before mapping of conllu to pytorch indices:")
    #print(edges_start)
    #print(edges_end)

    for idx, start_node in enumerate(edges_start.copy()):
      edges_start[idx] = conll_pytorch_idx_dict[start_node]
    for idx, end_node in enumerate(edges_end.copy()):
      edges_end[idx] = conll_pytorch_idx_dict[end_node]
    
    #print(f"Index dict{conll_pytorch_idx_dict}")
    #print("After mapping of conllu to pytorch indices:")
    #print(edges_start)
    #print(edges_end)
 
    min_end = utils.find_min(edges_start)
    max_end = utils.find_max(edges_end)
    if(min_end > max_end):
      node_indices = range(min_end, max_end)
    else:
      node_indices = range(min_end, max_end+1)

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
    # Add sub relation to edge features
    for node_idx in node_indices:
      word = words_graph[node_idx]
      tokens = tokenizer.tokenize(word)

      for token_idx,token in enumerate(tokens):
         # Replace first token
        if(token_idx == 0):
          #current_idx = node_idx +insertion_count
          current_idx = node_idx +insertion_count
          words_graph_tokenized[current_idx] = token
          #insertion_count = insertion_count+1
        # Add connection to first subword token
        elif(token_idx == 1):
          # add subword token to end of token list
          words_graph_tokenized.append( token)
          subword_token_idx = len(words_graph_tokenized) -1
          # Add sub dependency tag for subword token
          dependency_tags_sentence.append('sub')
          # add edge from last (sub)word token to current token
          edges_start_tokenized.append(current_idx)
          edges_end_tokenized.append(subword_token_idx)
          #edges_start_tokenized.append(subword_token_idx)
          #edges_end_tokenized.append(current_idx)

        # Add subword token after subword token
        else:
            words_graph_tokenized.append( token)
            subword_token_idx = len(words_graph_tokenized) -1
            edges_start_tokenized.append(subword_token_idx-1)
            edges_end_tokenized.append(subword_token_idx)
            #edges_start_tokenized.append(subword_token_idx)
            #edges_end_tokenized.append(subword_token_idx-1)

            # Add sub dependency tag for subword token
            dependency_tags_sentence.append('sub')

    # Convert graph tokens to ids
    ids_graph_tokenized = tokenizer.convert_tokens_to_ids(words_graph_tokenized)

    # Tokenize sentence with Bert tokenizer
    words_sentence_tokenized = []
    for word in raw_sentence.split(" "):
      #print(tokenizer.tokenize(word))
      words_sentence_tokenized.extend(tokenizer.tokenize(word))
    
    # If sentence and graph match: do not continue further processing
    if (len(words_graph_tokenized) == len(words_sentence_tokenized)+1):
      processed_sentences.append(raw_sentence)
    # Sentence and graph are not of same length (i.e. different tokenization): process further
    else:
      unresolved_sentences.append(sentence_idx)
      
      # Record unresolved sentence graph matchings
      count_graph_sentence_discrepancy = count_graph_sentence_discrepancy+1
      words_sentence_tokenized = []
      continue
      
    ids_sentence_tokenized = tokenizer.convert_tokens_to_ids(words_sentence_tokenized)

    # Map sentence token ids to graph token ids
    sentence_graph_idx_map = {}
    ids_graph_tokenized_temp = ids_graph_tokenized.copy()

    for idx, token_id in enumerate(ids_sentence_tokenized):
      if token_id in ids_graph_tokenized_temp:
        token_idx_graph = ids_graph_tokenized_temp.index(token_id)
        ids_graph_tokenized_temp[token_idx_graph] = "token processed"
        sentence_graph_idx_map[idx] = token_idx_graph
    
    tokens_graph = []
    # Get position of sentence tokens in graph 
    for idx, token_id in enumerate(ids_sentence_tokenized):
      if idx in sentence_graph_idx_map:
        graph_token_idx = ids_graph_tokenized[sentence_graph_idx_map[idx]]
        tokens_graph.append(tokenizer.convert_ids_to_tokens(graph_token_idx))
    
    # Create Pytorch data object
    edge_index = torch.tensor([edges_start_tokenized,edges_end_tokenized], dtype=torch.long)
    # Add edge attributes: dependency tags
    # One-Hot encode dependency tags in sentence
    oh_dependency_tags = oh_encoder_dependencies.transform(np.array(dependency_tags_sentence).reshape(-1,1)).toarray()
    # Duplicate dependency tags to create edge attributes list (undirected edges)
    # Changed to directed edges
    edge_attrs = [ i for i in oh_dependency_tags for r in range(1) ]
    edge_attrs = torch.tensor(np.array(edge_attrs), dtype=torch.float)

    # Add node attributes: sentence token ids
    ids_graph_tokenized_np = np.array(ids_graph_tokenized)

    # Pad to embedding size
    ids_graph_tokenized_padded = np.zeros((ids_graph_tokenized_np.shape[0], 768))
    for idx, token_id in enumerate(ids_graph_tokenized_np):
      ids_graph_tokenized_padded[idx][0] = token_id

    x = torch.tensor(ids_graph_tokenized_padded, dtype=torch.long)
    data = Data(x=x,edge_index=edge_index, edge_attr=edge_attrs)

    if (sentence_idx not in unresolved_sentences):
      syntax_graphs.append([data, sentence_graph_idx_map])
      if (len(words_graph_tokenized)-1 != len(words_sentence_tokenized)):
        print("Tokenized Sentences with wrong lengths:")
        print(words_graph_tokenized)
        print(words_sentence_tokenized)
        print("Sentences with wrong lengths:")
        print(words_graph)
        #print(words_sentence_processed)

    if( sentence_idx <= 7):
      save_pygeom_graph_image(data, filename.split(".")[0])
    
    if( print_graph ):
      #print(raw_sentence)

      save_pygeom_graph_image(data, filename.split(".")[0])
      print_graph = False
    
  print(f"Num syntax graphs created: {len(syntax_graphs)}")
  print(f"Num processed sentences: {len(processed_sentences)}")
  # Save processed corpus text
  filename_text = ud_file.split(".")[0] + f"-{tokenizer_name}.txt"
  filename_text = filename_text.replace("original/","")

  dirname = os.path.dirname(filename_text)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  
  with open(filename_text, 'w') as output:
    output.write("\n".join(processed_sentences))

  # Save list of Pytorch geometric data objects
  filename_syntree = filename_text.split(".")[0] + f"-{syntree_mode}.syntree"

  dirname = os.path.dirname(filename_syntree)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(filename_syntree, 'wb') as handle:
    pickle.dump(syntax_graphs, handle)
  
print(f"Ignored {count_graph_sentence_discrepancy} sentences because graph and sentence did not match")


  
