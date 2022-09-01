####
# Data preprocessing for constituency grammar treebank data
# Processes ICE-GB *.tre files
####

from sklearn import preprocessing
import numpy as np
import os
import glob

# Select number of threads to use
num_threads = "2"
os.environ["OMP_NUM_THREADS"] = num_threads # export OMP_NUM_THREADS=1
os.environ["OPENBLAS_NUM_THREADS"] = num_threads # export OPENBLAS_NUM_THREADS=1
os.environ["MKL_NUM_THREADS"] = num_threads # export MKL_NUM_THREADS=1
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads # export VECLIB_MAXIMUM_THREADS=1
os.environ["NUMEXPR_NUM_THREADS"] = num_threads # export NUMEXPR_NUM_THREADS=1


PID = os.getpid()
PGID = os.getpgid(PID)
print(f"PID: {PID}, PGID: {PGID}", flush=True)

data_path = "./data/original/ice-gb/"
# Set of syntactic constituency tags for ICE dataset
constituency_tags = []
# Raw text sentences
raw_sentences = []
# Processed text sentences
processed_sentences = []
# List of Pytorch Geometric syntax graphs
syntax_graphs = []


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
    root.add_children([Node(line) for line in text.splitlines()[0:100] if line.strip()])
    graphs = root.as_dict()['root']
    print(root.children)



  