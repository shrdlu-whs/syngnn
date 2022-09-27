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
import utilities_data_preprocessing as utils
import importlib
# Reload utils library if changed
importlib.reload(utils)

# Select number of threads to use
num_threads = "10"
os.environ["OMP_NUM_THREADS"] = num_threads # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = num_threads # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = num_threads # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = num_threads # export NUMEXPR_NUM_THREADS=1


PID = os.getpid()
PGID = os.getpgid(PID)
print(f"PID: {PID}, PGID: {PGID}", flush=True)

data_path = "./data/original/ice-gb/"
# BERT tokenizer to use:
tokenizer_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(tokenizer_name)

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


def const_tree_to_pytorch_geom(node, node_parent_id):
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
                # Add edge to token
                #edges_start.append(new_node_idx-1)
                #edges_end.append(new_node_idx)
        
    #conll_pytorch_idx_map.append(int(token.id))

    # Add edges from parent to current node
    #if node_id != 0:
    #    edges_start.append(node_parent_id)
    #    edges_end.append(node_id)

    node_parent_id = node_id
    for child_node in node.children:
        
        node_id = node_id+1
        # Add edges from parent to current node
        #if node_id != 0:
        edges_start.append(node_parent_id)
        edges_end.append(node_id)
        const_tree_to_pytorch_geom(child_node, node_parent_id)

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
        constituency_tags_set.append(constituency_tag)
        # Get constituency attributes
        if result.group(4) != None:
            constituency_attributes = result.group(4).split(",")
            constituency_attributes_set.extend(constituency_attributes)
        if result.group(6) != None:
            words = result.group(6)
            # Search for entities
            result = re.search(r"(&[a-z]+;)", words)
            if result != None:
                # Replace entity
                found_entity = result.group(1)
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
def create_pg_node_features(node_list, constituency_attributes_sentence, oh_encoder_constituency_tags, oh_encoder_constituency_attributes, embedding_size=768, num_const_attributes=113):
    node_index = []
    len_constituency_attributes_sentence = len(constituency_attributes_sentence)
    for node_idx, node in enumerate(node_list):
        # Node is constituency tag node
        if node_idx < len_constituency_attributes_sentence:
            oh_const_tag = oh_encoder_constituency_tags.transform(np.array(node).reshape(-1, 1)).toarray()
            oh_const_attributes = np.zeros(num_const_attributes,dtype=int)
            for const_attr in constituency_attributes_sentence[node_idx]:
                #print(const_attr)
                oh_const_attr = oh_encoder_constituency_attributes.transform(np.array(const_attr).reshape(-1, 1)).toarray()
                #print(oh_const_attr)
                oh_index = np.where(oh_const_attr[0] == 1)
                oh_const_attributes[oh_index] = 1

            # Append one hot encoded const tag and attributes
            node_embedding = []
            #print(oh_const_tag)
            #print(oh_const_attributes)
            node_embedding.extend(oh_const_tag[0])
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

for filename in glob.iglob(data_path + '**/written_*.tre', recursive=True):
  with open(filename, encoding='cp1252') as ice_file:
    syntax_graphs = []
    raw_sentences = []
    ice_filepath = os.path.abspath(filename)
    ice_filename = os.path.basename(ice_filepath)
    print(ice_filename)
    text = ice_file.read()
    root = Node('root')
    root.add_children([Node(line) for line in text.splitlines() if line.strip()])
    #graphs = root.as_dict()
    #print(graphs)
    entity_set = list(set(entity_set))
    #print(len(entity_set))
    print(entity_set)
    constituency_tags_set = list(set(constituency_tags_set))
    #print(constituency_tags_set)
    print("Constituency tags:")
    print(len(constituency_tags_set))
    constituency_attributes_set = list(set(constituency_attributes_set))
    num_const_attributes = len(constituency_attributes_set)
    print("Constituency attributes:")
    print(num_const_attributes)
    #print(constituency_attributes_set)
    
    # Create one-hot encoder for constituency tags and attributes
    oh_encoder_constituency_tags = preprocessing.OneHotEncoder(dtype=int)
    oh_encoder_constituency_tags.fit(np.array(constituency_tags_set).reshape(-1, 1))
    oh_encoder_constituency_attributes = preprocessing.OneHotEncoder(dtype=int)
    oh_encoder_constituency_attributes.fit(np.array(constituency_attributes_set).reshape(-1, 1))
    
    sentences = root.children

    for sentence in sentences:
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
        const_tree_to_pytorch_geom(sentence, node_parent_id)
        '''print("Extracted graph")
        print(words_sentence)
        print(constituency_tags_sentence)
        print(constituency_attributes_sentence)
        print(edges_start)
        print(edges_end)
        print(tokens_graph)'''

        node_list, sentence_graph_idx_map, edges_start, edges_end = add_tokens_to_graph(constituency_tags_sentence, tokens_graph, edges_start, edges_end)
        '''print("--------------")
        print(node_list)
        print(sentence_graph_idx_map)
        print(edges_start)
        print(edges_end)'''
        #print(sentence_graph_idx_map)
        node_index = create_pg_node_features(node_list, constituency_attributes_sentence, oh_encoder_constituency_tags, oh_encoder_constituency_attributes, num_const_attributes=num_const_attributes)
        #print(node_index[-1])

        # Create Pytorch data object
        edge_index = torch.tensor([edges_start,edges_end], dtype=torch.long)
        x = torch.tensor(np.array(node_index), dtype=torch.long)
        data = Data(x=x,edge_index=edge_index)
        #print(data)

        syntax_graphs.append([data, sentence_graph_idx_map])
        raw_sentences.append(" ".join(words_sentence))
        #print(words_sentence)
        #print(tokens_graph)
    #print(ice_filepath)
    filename_text = ice_filepath.split(".")[0]
    filename_text = filename_text.replace("original/","")
    filename_text_train = filename_text + f"-train-{tokenizer_name}.txt"
    filename_text_dev = filename_text + f"-dev-{tokenizer_name}.txt"
    filename_text_test = filename_text + f"-test-{tokenizer_name}.txt"
    print(filename_text_train)

    # Split sentences in train, dev, test
    sentences_train, sentences_test = train_test_split(raw_sentences, test_size=0.25, shuffle=True, random_state=42)
    sentences_dev, sentences_test = train_test_split(sentences_test,test_size=0.5, shuffle=True, random_state=42)
    # Save train, dev, test sentences
    utils.save_sentences(sentences_train, filename_text_train)
    utils.save_sentences(sentences_dev, filename_text_dev)
    utils.save_sentences(sentences_test, filename_text_test)

    # Save list of Pytorch geometric data objects
    #filename_syntree = filename_text.split(".")[0] + f".syntree"
    filename_syntree_train = filename_text + f"-train-{tokenizer_name}.syntree"
    filename_syntree_dev = filename_text + f"-dev-{tokenizer_name}.syntree"
    filename_syntree_test = filename_text + f"-test-{tokenizer_name}.syntree"
    
    print(filename_syntree_train)
    # Split syntax graphs in train, dev, test
    syntax_graphs_train, syntax_graphs_test = train_test_split(syntax_graphs, test_size=0.25, shuffle=True, random_state=42)
    syntax_graphs_dev, syntax_graphs_test = train_test_split(syntax_graphs_test,test_size=0.5, shuffle=True, random_state=42)
    # Save train, dev, test sentences
    utils.save_syntrees(syntax_graphs_train, filename_syntree_train)
    utils.save_syntrees(syntax_graphs_dev, filename_syntree_dev)
    utils.save_syntrees(syntax_graphs_test, filename_syntree_test)
    #dirname = os.path.dirname(filename_syntree_train)
    #if not os.path.exists(dirname):
    #    os.makedirs(dirname)

    #with open(filename_syntree, 'wb') as handle:
    #    pickle.dump(syntax_graphs, handle)

    print(f"Saved {len(raw_sentences)} sentences and {len(syntax_graphs)} graphs")





  