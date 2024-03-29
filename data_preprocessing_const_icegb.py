####
# Data preprocessing for constituency grammar treebank data
# Processes ICE-GB *.tre files
####

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import numpy as np
import os
import glob
import re
from transformers import BertTokenizer
import torch
from torch_geometric.data import Data
import torch_geometric.utils as tg_utils
import spacy
from spacy import displacy
import utilities_data_preprocessing as utils
import itertools
import importlib
# Reload utils library if changed
importlib.reload(utils)

# Select number of threads to use
num_threads = "14"
os.environ["OMP_NUM_THREADS"] = num_threads # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = num_threads # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = num_threads # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = num_threads # export NUMEXPR_NUM_THREADS=1


PID = os.getpid()
PGID = os.getpgid(PID)
print(f"PID: {PID}, PGID: {PGID}", flush=True)

# BERT tokenizer to use:
tokenizer_name = 'bert-base-cased'
print(f"Tokenizer: {tokenizer_name}")
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

# build the Pytorch geometric graphs from gold standard hand-annotated syntax trees
mode = "GOLD"
# Generate syntax trees for text files automatically with Spacy and Berkeley Nueral Parser
#mode = "GEN"

if mode == "GOLD":
    data_path = "./data_sample/original/ice-gb/"
    data_path = data_path + '**/*.tre'
    encoding = 'cp1252'
    # Number of lines or -1 for all lines
    num_lines = -1
elif mode == "GEN":
    data_path = "./data_sample/ice-gb/"
    data_path = data_path + f"**/*-gold-*-{tokenizer_name}.txt"
    encoding = 'utf-8'
    # Number of lines or -1 for all lines
    num_lines = -1
    # Load the language model
    nlp = spacy.load("en_core_web_lg")
    import benepar
    benepar.download('benepar_en3')
    if spacy.__version__.startswith('2'):
        nlp.add_pipe(benepar.BeneparComponent("benepar_en3"))
    else:
        nlp.add_pipe("benepar", config={"model": "benepar_en3"})

print(f"Mode: {mode}")
print(data_path)

entities = {
    "&semi;":";",
    "&lsquo;":"'",
    "&rsquo;":"'",
    "&ldquo;":"“",
    "&rdquo;":"”",
    "&oumlaut;":"ö",
    "&aumlaut;":"ä",
    "&uumlaut;":"ü",
    "&Oumlaut;":"Ö",
    "&Aumlaut;":"Ä",
    "&Uumlaut;":"Ü",
    "&aacute;":"á",
    "&Aacute;":"Á",
    "&agrave;":"à",
    "&Agrave;":"À",
    "&eacute;":"é",
    "&Eacute;":"É",
    "&ecircumflex;":"ê",
    "&egrave;":"è",
    "&ccedille;":"ç",
    "&Ccedille;":"Ç",
    "&oeligature;":"œ",
    "&OEligature;":"Œ",
    "&ampersand;":"&",
    "&alpha;":"α",
    "&beta;":"β",
    "&gamma;":"γ",
    "&delta":"δ",
    "&epsilon":"ε",
    "&mu;":"μ",
    "&omega":"ω",
    "&degree;":"°",
    "&bullet;":"-",
    "&arrow;":"→",
    "&arrowhead;":">",
    "&square;":"□",
    "&dagger;":"†",
    "&star;":"⋆",
    "&dot;":"·"
    }
# Set of syntactic constituency tags for ICE dataset
constituency_tags = []
# Raw text sentences
raw_sentences = []
# Processed text sentences
processed_sentences = []
# List of Pytorch Geometric syntax graphs
syntax_graphs = []

node_id = 0
node_parent_id = 0
constituency_tags_sentence = []
constituency_attributes_sentence = []
words_sentence = []
words_graph = []
tokens_graph = {}
# Graph edges
edges_start = []
edges_end = []
# Edges from constituency tag to const tag
edges_start_const_const = []
edges_end_const_const = []
# Edges from constituency tag to token
edges_start_const_token = []
edges_end_const_token = []
# Edges from token to token
edges_start_token_token = []
edges_end_token_token = []

# %%
def const_tree_gold_to_pytorch_geom(node, node_parent_id):
    global node_id
    # Get words
    words = node.words
    # Add node constituency relation to list     
    constituency_tags_sentence.append(node.constituency_tag)
    # Add constituency attributes if given
    if node.constituency_attributes != None:
        constituency_attributes_sentence.append(node.constituency_attributes)
    else:
        constituency_attributes_sentence.append([])
    if words != None:
        tokens_graph[node_id] = []
        words_sentence.extend(words)
        # Tokenize words and add tokens to dict
        for word_idx, word in enumerate(words):
            tokens = tokenizer.tokenize(word)
            for token_idx, token in enumerate(tokens):
                tokens_graph[node_id].append(tokenizer.convert_tokens_to_ids(token))

    node_parent_id = node_id
    for child_node in node.children:
        
        node_id = node_id+1
        # Add edges from parent to current node
        edges_start.append(node_parent_id)
        edges_end.append(node_id)
        const_tree_gold_to_pytorch_geom(child_node, node_parent_id)

# %%
def const_tree_gen_to_pytorch_geom(node, start_node_id):
    global node_id
    #print(node.text)
    #print(start_node_id)
    #print(node._.labels)
    #print(node.text)
    # Node is leaf node
    if len(node._.labels) == 0:
        words = [node.text]
        # Get POs tag of word
        constituency_tags_sentence.append(node[0].tag_)
        #print(node[0].tag_)

    else:
        # Node is constituency tag node
        constituency_tags_sentence.append(node._.labels[0])
        words = []
        # Constituency node with only one child, e.g. NP->PRP
        # Extract word and POS tag
        if len(node) == 1 and hasattr(node[0],'tag_'):
            constituency_tags_sentence.append(node[0].tag_)
            words = [node.text]
            node_id = node_id+1
            edges_start.append(start_node_id)
            edges_end.append(node_id)
    
    if words != None:
        tokens_graph[node_id] = []
        words_sentence.extend(words)
        # Tokenize words and add tokens to dict
        for word_idx, word in enumerate(words):
            tokens = tokenizer.tokenize(word)
            for token_idx, token in enumerate(tokens):
                tokens_graph[node_id].append(tokenizer.convert_tokens_to_ids(token))

    #node_parent_id = node_id
    #start_node_id = node_id
    for child_node in node._.children:
        #print(start_node_id)
        #print(child_node._.labels)
        #print(child_node.text)
        #print(node._.labels)
        node_id = node_id+1
        # Add edges from parent to current node
        #edges_start.append(node_parent_id)
        edges_start.append(start_node_id)
        edges_end.append(node_id)

        const_tree_gen_to_pytorch_geom(child_node, node_id)
# %%
entity_set = []
constituency_tags_set = []
constituency_attributes_set = []

def analyse_line(line):
    # Ignore ICE identification line
    if line.startswith("<ICE-GB:"):
        return None, None, None
    if line.startswith("root"):
        return "root", None, None
    words = None
    constituency_tag = None
    constituency_attributes = None
    result = re.search(r"([A-Z]+),([A-Z]+)(\(([a-z\,]+)\))?( {(.+)})?", line)
    if result != None:
        # Get constituency tag
        constituency_tag = result.group(2)
        if constituency_tag not in constituency_tags_set:
            constituency_tags_set.append(constituency_tag)
        # Get constituency attributes
        if result.group(4) != None:
            constituency_attributes = result.group(4).split(",")
            # Item marked as ignore: discard line
            if "ignore" in constituency_attributes:
                return None, None, None
            for const_attr in constituency_attributes:
                if const_attr not in constituency_attributes_set:
                    constituency_attributes_set.append(const_attr)
        if result.group(6) != None:
            words = result.group(6)
            # Search for entities
            result = re.search(r"(&[a-z]+;)", words)
            if result != None:
                # Replace entity
                found_entity = result.group(1)
                if found_entity not in entity_set:
                    entity_set.append(found_entity)
                if found_entity in entities.keys():
                    words = words.replace(found_entity, entities[found_entity])
            words = words.replace("<l>","")
            words = words.replace("<l->","-")

            words = words.split(" ")
    #print(f"{constituency_tag} -{words}")
    return constituency_tag, constituency_attributes, words

# Add nodes for Bert tokens to syntax graph
def add_tokens_to_graph(constituency_tags, tokens_graph, edges_start, edges_end):
    node_list = constituency_tags.copy()
    sentence_graph_token_map = []
    for key in tokens_graph.keys():
        # Add tokens as child nodes of key node
        for token_idx, token in enumerate(tokens_graph[key]):
            node_list.append(token)
            node_idx = len(node_list)-1
            sentence_graph_token_map.append(node_idx)
            if token_idx == 0:
                edges_start.append(key)
                edges_end.append(node_idx)
            else:
                edges_start.append(node_idx-1)
                edges_end.append(node_idx)
    sentence_graph_token_map = {list_idx: token_idx for list_idx,token_idx in enumerate(sentence_graph_token_map)}
    return node_list, sentence_graph_token_map, edges_start, edges_end

# Create Pytorch geometric node features of size num_nodes*embedding_size
# For Bert tokens: placeholder of zeroes for real Bert embeddings added by SynGNN
# For constituency tree nodes: concatenated one-hot encodings of constituency tag and constituency attributes of the node with padding zeroes to fill 768
def create_pg_node_features(node_list, num_const_graph_nodes, oh_encoder_constituency_tags, constituency_attributes_sentence=None, oh_encoder_constituency_attributes=None, embedding_size=768, num_const_attributes=113):
    node_index = []
    #num_const_graph_nodes = len(constituency_attributes_sentence)
    for node_idx, node in enumerate(node_list):
        # Node is constituency tag node
        if node_idx < num_const_graph_nodes:
            # Append one hot encoded const tag and attributes
            node_embedding = []
            oh_const_tag = oh_encoder_constituency_tags.transform(np.array(node).reshape(-1, 1)).toarray()
            node_embedding.extend(oh_const_tag[0])
            if constituency_attributes_sentence != None:
                oh_const_attributes = np.zeros(num_const_attributes,dtype=int)
                for const_attr in constituency_attributes_sentence[node_idx]:
                    #print(const_attr)
                    oh_const_attr = oh_encoder_constituency_attributes.transform(np.array(const_attr).reshape(-1, 1)).toarray()
                    #print(oh_const_attr)
                    oh_index = np.where(oh_const_attr[0] == 1)
                    oh_const_attributes[oh_index] = 1
                node_embedding.extend(oh_const_attributes)

            padding_length = embedding_size-len(node_embedding)
            for i in range(padding_length):
                node_embedding.append(0)
            node_index.append(node_embedding)

        # Node is token node
        else:
            node_index.append(np.zeros(embedding_size,dtype=int))
    return node_index

class Node:
    def __init__(self, indented_line):
        self.children = []
        self.level = len(indented_line) - len(indented_line.lstrip())
        self.text = indented_line.strip()
        self.constituency_tag, self.constituency_attributes, self.words = analyse_line(self.text)

    def add_children(self, nodes):
        childlevel = nodes[0].level
        while nodes:
            node = nodes.pop(0)
            # Node is sentence beginning
            if node.constituency_tag != None:

                if node.level == childlevel: # add node as a child
                    self.children.append(node)
                elif node.level > childlevel: # add nodes as grandchildren of the last child
                    nodes.insert(0,node)
                    self.children[-1].add_children(nodes)
                elif node.level <= self.level: # this node is a sibling, no more children
                    nodes.insert(0,node)
                    return

    def as_dict(self):
        if self.words == None:
            key = self.constituency_tag
        else:
            key = f"{self.constituency_tag} - {self.words}"
        if len(self.children) > 1:
            return {key: [node.as_dict() for node in self.children]}
        elif len(self.children) == 1:
            return {key: self.children[0].as_dict()}
        else:
            return key

# %%
# Load networkx
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
# Load matplotlib.pyplot
import matplotlib.pyplot as plt
def create_networkx_node_attributes(node_attributes, node_idx_list,encoder=None):
  node_attrs_networkx = {}
  #print(node_attributes)
  for idx, node_attr in enumerate(node_attributes):
    # Get node index from list
    node_idx = node_idx_list[idx]
    # Add label to list
    if isinstance(node_attr,int):
        node_label = tokenizer.convert_ids_to_tokens([node_attr])[0]
    elif isinstance(node_attr,str):
       
        node_label = node_attr
        #print(node_label)
    node_attrs_networkx[node_idx] = node_label

  return node_attrs_networkx

# Save Pytorch Geometric Data object as png image
def save_pygeom_graph_image(data, filename, sentence_idx,node_attr_list=None):

    sentenceIdx = sentence_idx + 1

    graph = tg_utils.to_networkx(data)
    # Create depth-first tree from graph
    #graph = nx.dfs_tree(graph, source=0)
    idx_order = list(graph.nodes)
    # Create networkx node labels with tokens
    node_attrs_networkx = create_networkx_node_attributes(node_attr_list, idx_order)

    dirname = os.path.dirname("./images/const_graphs/")
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

    #nx.draw_networkx_edge_labels(graph, pos = layout, edge_labels=edge_attrs_networkx, font_size=8)
    plt.savefig(filepath_png)
    plt.clf()
    # Remove dot file
    os.remove(filepath_dot)
    print(filepath_png)
# %%
# Create One-Hot Encoders from data
for filename in glob.iglob(data_path, recursive=True):

  with open(filename, encoding=encoding) as ice_file:
    ice_filepath = os.path.abspath(filename)
    ice_filename = os.path.basename(ice_filepath)
    #print(ice_filename)
    text = ice_file.read()
    
    #print(text[-100:-1])
    #text = text +"\n"
    
    if mode == "GOLD":
        text = text.splitlines()[0:num_lines]
        root = Node('root')
        root.add_children([Node(line) for line in text if line.strip()])
        sentences = root.children
        #entity_set = list(set(entity_set))
        #constituency_tags_set = list(set(constituency_tags_set))

        #constituency_attributes_set = list(set(constituency_attributes_set))
        del(text)
    elif mode == "GEN":
        sentences = text.splitlines()

        del(text)
        for sent_idx, sentence in enumerate(sentences):
            node_id = 0
            node_parent_id = 0
            constituency_tags_sentence = []
            constituency_attributes_sentence = []
            words_sentence = []
            tokens_sentence = []
            tokens_graph = {}
            word_indices_sentence = []
            token_indices_sentence = []
            # Graph edges
            edges_start = []
            edges_end = []

            doc = nlp(sentence)
            sentence = list(doc.sents)[0]

            const_tree_gen_to_pytorch_geom(sentence, node_parent_id)
            for const_tag in constituency_tags_sentence:
                if const_tag not in constituency_tags_set:
                    constituency_tags_set.append(const_tag)
if mode == "GOLD":
    print(constituency_attributes_set)
    num_const_attributes = len(constituency_attributes_set)
    oh_encoder_constituency_attributes = preprocessing.OneHotEncoder(dtype=int)
    oh_encoder_constituency_attributes.fit(np.array(constituency_attributes_set).reshape(-1, 1))
    
oh_encoder_constituency_tags = preprocessing.OneHotEncoder(dtype=int)
oh_encoder_constituency_tags.fit(np.array(constituency_tags_set).reshape(-1, 1))
print(constituency_tags_set)


# %%
# Create syntax trees and text files
for filename in glob.iglob(data_path, recursive=True):

  with open(filename, encoding=encoding) as ice_file:
    syntax_graphs = []
    raw_sentences = []
    ice_filepath = os.path.abspath(filename)
    ice_filename = os.path.basename(ice_filepath)
    print(ice_filename)
    text = ice_file.read()
    
    if mode == "GOLD":
        text = text.splitlines()[0:num_lines]
        root = Node('root')
        root.add_children([Node(line) for line in text if line.strip()])
        sentences = root.children

    elif mode == "GEN":
        sentences = text.splitlines()

    del(text)
    
    for sent_idx, sentence in enumerate(sentences):
        #for child in sentence.children:
        #    print(child.constituency_tag)
        node_id = 0
        node_parent_id = 0
        constituency_tags_sentence = []
        constituency_attributes_sentence = []
        words_sentence = []
        tokens_sentence = []
        tokens_graph = {}
        word_indices_sentence = []
        token_indices_sentence = []
        # Graph edges
        edges_start = []
        edges_end = []
        if mode == "GOLD":
            const_tree_gold_to_pytorch_geom(sentence, node_parent_id)
        else:
            #print(sentence)
            # Generate syntax trees from raw sentence4
            doc = nlp(sentence)
            sentence = list(doc.sents)[0]
            if sent_idx == len(sentences)-1:
                print("Parse string")
                print(sentence._.parse_string)
            
            const_tree_gen_to_pytorch_geom(sentence, node_parent_id)
        
        '''print("Extracted graph")
        print(words_sentence)
        print(constituency_tags_sentence)
        print(constituency_attributes_sentence)
        print(edges_start)
        print(edges_end)
        print(tokens_graph)
        print(constituency_tags_sentence)'''

        num_const_graph_nodes = len(constituency_tags_sentence)
        node_list, sentence_graph_idx_map, edges_start, edges_end = add_tokens_to_graph(constituency_tags_sentence, tokens_graph, edges_start, edges_end)
        print("--------------")
        print(node_list)
        print(sentence_graph_idx_map)
        print(edges_start)
        print(edges_end)
        print(sentence_graph_idx_map)
        # For gold constituency tags: add constituency attributes to node features
        # Optional, standard gold constituency trees consist of const_labels only
        #if mode == "GOLD":
        #    node_index = create_pg_node_features(node_list, num_const_graph_nodes,  oh_encoder_constituency_tags, constituency_attributes_sentence, oh_encoder_constituency_attributes, num_const_attributes=num_const_attributes)
        #else:
        node_index = create_pg_node_features(node_list, num_const_graph_nodes,  oh_encoder_constituency_tags)
        #print(node_index[-1])

        # Create Pytorch data object
        edge_index = torch.tensor([edges_start,edges_end], dtype=torch.long)
        x = torch.tensor(np.array(node_index), dtype=torch.long)
        data = Data(x=x,edge_index=edge_index)
        
        if (sent_idx <=25):
            mode_name = mode.lower()
            filename = ice_filename+f"-{mode_name}"
            save_pygeom_graph_image(data,filename,sent_idx,node_list)

        
        #print(tokens_graph)
    #print(ice_filepath)

    # Save list of Pytorch geometric data objects
    filename = ice_filepath.split(".")[0]
    filename = filename.replace("original/","")
    mode_name = mode.lower()



    if len(sentence_graph_idx_map) == 0:
            print("No tokens found")
            print(data)
            print(words_sentence)
    else:

        syntax_graphs.append([data, sentence_graph_idx_map])
        raw_sentences.append(" ".join(words_sentence))

    if mode == "GOLD":
        filename_text_train = filename + f"-train-{tokenizer_name}.txt"
        filename_text_dev = filename + f"-dev-{tokenizer_name}.txt"
        filename_text_test = filename + f"-test-{tokenizer_name}.txt"

        
        # Split sentences in train, dev, test
        sentences_train, sentences_test = train_test_split(raw_sentences, test_size=0.25, shuffle=True, random_state=42)
        sentences_dev, sentences_test = train_test_split(sentences_test,test_size=0.5, shuffle=True, random_state=42)
        # Save train, dev, test sentences
        utils.save_sentences(sentences_train, filename_text_train)
        utils.save_sentences(sentences_dev, filename_text_dev)
        utils.save_sentences(sentences_test, filename_text_test)

        print(filename_text_train)
        filename_syntree_train = filename + f"-train-{tokenizer_name}-{mode_name}.syntree"
        filename_syntree_dev = filename + f"-dev-{tokenizer_name}-{mode_name}.syntree"
        filename_syntree_test = filename + f"-test-{tokenizer_name}-{mode_name}.syntree"

        # Split syntax graphs in train, dev, test
        syntax_graphs_train, syntax_graphs_test = train_test_split(syntax_graphs, test_size=0.25, shuffle=True, random_state=42)
        syntax_graphs_dev, syntax_graphs_test = train_test_split(syntax_graphs_test,test_size=0.5, shuffle=True, random_state=42)
        # Save train, dev, test syntrees
        utils.save_syntrees(syntax_graphs_train, filename_syntree_train)
        utils.save_syntrees(syntax_graphs_dev, filename_syntree_dev)
        utils.save_syntrees(syntax_graphs_test, filename_syntree_test)

        print(filename_syntree_train)
    else:
        filename_syntree = filename + f"-{mode_name}.syntree"
        utils.save_syntrees(syntax_graphs, filename_syntree)
        print(filename)



    print(f"Saved {len(raw_sentences)} sentences and {len(syntax_graphs)} graphs")





  