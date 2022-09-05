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
from nltk.tokenize import word_tokenize
nltk.download('punkt')
import utilities_data_preprocessing as utils

# Reload utils library if changed
importlib.reload(utils)

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
data_path_dev = "./data/original/ud/UD_English-GUM/"
data_path = "./data/original/ud/UD_English-GUM"
# BERT tokenizer to use:
tokenizer_name = 'bert-base-cased'
# Set of syntactic dependency tags
dependency_tags = ["-","sub","root","punct","dep","nsubj","nsubj:pass","nsubj:outer","obj","iobj","csubj","csubj:pass","csubj:outer","ccomp","xcomp","nummod","appos","nmod","nmod:npmod","nmod:tmod","nmod:poss","acl","acl:relcl","amod","det","det:predet","case","obl","obl:npmod","obl:tmod","advcl","advmod","compound","compound:prt","fixed","flat","flat:foreign","goeswith","vocative","discourse","expl","aux","aux:pass","cop","mark","conj","cc","cc:preconj","parataxis","list","dislocated","orphan","reparandum", "obl:agent"]


device =  torch.device('cpu')

tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
print_graph = False
count_graph_sentence_discrepancy = 0

def dep_tree_to_pytorch_geom(tree):

    # Get data of root node
    token = tree.data
    # Add node dependency relation to list     
    dependency_tags_sentence.append(token.deprel)

    words_graph.append(token.form)
    conll_pytorch_idx_map.append(int(token.id))

    # Add edges from parent to current node
    if( int(token.id) != 0):
        edges_start.append(int(token.head))
        edges_end.append(int(token.id))
        #edges_start.append(int(token.id))
        #edges_end.append(int(token.head))

    #print(f"{token.head}->{token.id}, {token.upos}, {token.form}, {token.deprel}")

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
      #print(words_sentence_tokenized)

    #if (raw_sentence.find("Finally, findings on enjambment") != -1):
    #  print(words_graph_tokenized)
    #  print(words_sentence_tokenized)
    
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
      ###############################################################
      # Start graph to sentence alignment
      # Process raw sentence
      # Align split words in graph (e.g. negative modals) with raw sentence - for example convert wasn't to was n't
      words_sentence = word_tokenize(raw_sentence)
      #words_sentence = raw_sentence.split(" ")
      words_sentence_temp = words_sentence.copy()

      insertion_count = 0
      for word_idx, word in enumerate(words_sentence):

        if (word == "''" or word == "``"):
          words_sentence_temp[word_idx] = '"'
        if (word.find("-",1) == -1 or word == "--"):
          continue
  
        # Check if word exists in graph
        if word in words_graph:
            continue
        words_sentence_temp.pop(word_idx+insertion_count)
        insertion_count = insertion_count-1
        split_word = word.split("-")

        for sub_idx, substring in enumerate(split_word):
            words_sentence_temp.insert(word_idx+insertion_count, substring)
            insertion_count = insertion_count+1
            if (sub_idx != len(split_word)-1):
              words_sentence_temp.insert(word_idx+insertion_count, "-")
              insertion_count = insertion_count+1

      # Copy graph words w/o root node
      words_graph_temp = words_graph.copy()[1:]
      # Copy tokenized sentences
      words_sentence_processed = words_sentence_temp.copy()
      
      words_sentence_temp, words_graph_temp, remaining_tokens_sentence_idx, remaining_tokens_graph_idx = utils.compare_sentence_to_graph(words_sentence_temp, words_graph_temp)
      joined_strings_sentence, joined_strings_sentence_index_list = utils.join_consecutive_tokens(words_sentence_temp, remaining_tokens_sentence_idx)
      joined_strings_graph, joined_strings_graph_index_list = utils.join_consecutive_tokens(words_graph_temp, remaining_tokens_graph_idx)

      insertion_count = 0 
      for list_idx_sentence, joined_string_sentence in enumerate(joined_strings_sentence):

        if ( sentence_idx == 531):
          print("looping through joined strings")

        # Check if joined string exists in graph
        try:
          list_idx_graph = joined_strings_graph.index(joined_string_sentence)
        except:
          continue

        if (list_idx_graph >=0):
          joined_string_graph = joined_strings_graph[list_idx_graph]
          onestring_indices_graph = joined_strings_graph_index_list[list_idx_graph]
          onestring_indices_sentence = joined_strings_sentence_index_list[list_idx_sentence]

          # get position for delete and insert in words_sentence_temp
          position_delete = onestring_indices_sentence[0]
          len_onestring_indices_sentence = len(onestring_indices_sentence)
          for pos_idx in range(len_onestring_indices_sentence):
            try:
              words_sentence_temp.pop(position_delete)
              words_sentence_processed.pop(position_delete)
            except:
              #print(f"Index error at: {raw_sentence}. Ignoring sentence.")
              break

            joined_strings_sentence_index_list = utils.shift_token_indices_in_list_of_index_lists( joined_strings_sentence_index_list, position_delete, -1)

          position_insert = position_delete
          for pos_idx in onestring_indices_graph:
            graph_token = words_graph_temp[pos_idx]
            words_sentence_temp.insert(position_insert, graph_token)
            words_sentence_processed.insert(position_insert, graph_token)
            joined_strings_sentence_index_list = utils.shift_token_indices_in_list_of_index_lists( joined_strings_sentence_index_list, position_insert, 1)
            position_insert = position_insert +1
          
      # Re-calculate amount of unmatching sentences and graphs
      words_sentence_temp, words_graph_temp, remaining_tokens_sentence_idx, remaining_tokens_graph_idx = utils.compare_sentence_to_graph(words_sentence_temp, words_graph_temp)
      if (len(set(words_graph_temp))>1 or len(set(words_sentence_temp))>1):
        unresolved_sentences.append(sentence_idx)
      
        # Record unresolved sentence graph matchings
        count_graph_sentence_discrepancy = count_graph_sentence_discrepancy+1

      else:
        # Add final graph-aligned sentence to processed sentences
        processed_sentences.append(" ".join(words_sentence_processed))

        # Tokenize sentence
        for word in words_sentence_processed:
          tokens = tokenizer.tokenize(word)
        for token_idx,token in enumerate(tokens):
            words_sentence_tokenized.append(token)


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
    #if (sentence_idx not in unresolved_sentences):
    if (sentence_idx not in unresolved_sentences):
      syntax_graphs.append([data, sentence_graph_idx_map])
      if (len(words_graph_tokenized)-1 != len(words_sentence_tokenized)):
        print("Tokenized Sentences with wrong lengths:")
        print(words_graph_tokenized)
        print(words_sentence_tokenized)
        print("Sentences with wrong lengths:")
        print(words_graph)
        #print(words_sentence_processed)

    if( sentence_idx <= 5):
      save_pygeom_graph_image(data, filename.split(".")[0])

    if (sentence_graph_idx_map == {0: 15, 1: 3, 2: 2, 3: 5, 4: 6, 5: 4, 6: 8, 7: 10, 8: 9, 9: 7, 10: 89, 11: 11, 12: 12, 13: 13, 14: 90, 15: 1, 16: 16, 17: 14, 18: 18, 19: 17, 20: 20, 21: 21, 22: 22, 23: 19, 24: 24, 25: 25, 26: 23, 27: 27, 28: 26, 29: 29, 30: 30, 31: 28, 32: 32, 33: 33, 34: 34, 35: 31, 36: 36, 37: 37, 38: 38, 39: 39, 40: 35, 41: 41, 42: 40, 43: 88, 44: 42, 45: 44, 46: 45, 47: 47, 48: 46, 49: 48, 50: 43, 51: 50, 52: 49, 53: 52, 54: 53, 55: 54, 56: 51, 57: 56, 58: 57, 59: 55, 60: 59, 61: 60, 62: 58, 63: 92, 64: 93, 65: 94, 66: 62, 67: 61, 68: 95, 69: 65, 70: 64, 71: 67, 72: 66, 73: 68, 74: 63, 75: 96, 76: 70, 77: 71, 78: 72, 79: 69, 80: 73, 81: 81, 82: 80, 83: 75, 84: 84, 85: 85, 86: 82, 87: 79, 88: 86, 89: 83, 90: 76, 91: 77, 92: 74, 93: 87, 94: 78, 95: 91}):
        print("Sentence for mapping:")
        print(words_graph_tokenized)
        print(words_sentence_tokenized)

    """if (raw_sentence.find("Finally, findings on enjambment") != -1):
      print_graph=True
      print(words_sentence_processed)
      print(words_sentence_temp)
      print(words_graph)
      print(words_graph_tokenized)
      print(tokenizer.tokenize("diachronic"))
      test = ['root', 'discussed', 'Finally', ',', 'findings', 'enjambment', 'on', 'corpus', 'in', 'our', 'diachronic', 'sonnet', 'are', '.']
      tokenized = []
      for word in test:
        tokenized_word = tokenizer.tokenize(word)
        tokenized.extend(tokenized_word)
      print(tokenized)"""
    
    if( print_graph ):
      #print(raw_sentence)

      save_pygeom_graph_image(data, filename.split(".")[0])
      print_graph = False
    



 
  print(f"Num syntax graphs created: {len(syntax_graphs)}")
  print(f"Num processed sentences: {len(processed_sentences)}")
  # Save processed corpus text
  filename_text = ud_file.split(".")[0] + f".txt"
  filename_text = filename_text.replace("original/","")

  dirname = os.path.dirname(filename_text)
  if not os.path.exists(dirname):
    os.makedirs(dirname)
  
  with open(filename_text, 'w') as output:
    output.write("\n".join(processed_sentences))

  # Save list of Pytorch geometric data objects
  filename_syntree = filename_text.split(".")[0] + f"-{tokenizer_name}.syntree"

  dirname = os.path.dirname(filename_syntree)
  if not os.path.exists(dirname):
    os.makedirs(dirname)

  with open(filename_syntree, 'wb') as handle:
    #print(filename_syntree)
    #print(syntax_graphs[0:5])
    #print(ids_graph_tokenized_np)
    #print(len(syntax_graphs))
    pickle.dump(syntax_graphs, handle)
  
print(f"Ignored {count_graph_sentence_discrepancy} sentences because graph and sentence did not match")


  
