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
import numpy as np
import os
import pickle

# Replace tokens in sentence
def join_consecutive_tokens(token_array, idx_list):
    """
    returns: list of tuples (joined token string, list of original idxs)
    """
    token_strings = []
    token_string_index_list = []
    token_string = ""
    token_string_index_list_temp = []

    if (len(token_array) == 0 and len(idx_list) == 0):
        return token_strings

    for count_idx, token_idx in enumerate(idx_list):
                # End of list append remaining string
        # Always add first token
        if (count_idx == 0 ):
            token_string = "".join([token_string,token_array[token_idx]])
            token_string_index_list_temp.append(token_idx)

        # if token_idx is consecutive: join strings
        elif (count_idx > 0 and idx_list[count_idx-1] == token_idx-1):
            token_string = "".join([token_string,token_array[token_idx]])
            token_string_index_list_temp.append(token_idx)

        #End of list append remaining string
        if (count_idx == len(idx_list)-1):
            token_strings.append(token_string)
            token_string_index_list.append(token_string_index_list_temp)

        # token_idx not consecutive: save found string and start new token string
        elif (count_idx > 0 and idx_list[count_idx-1] != token_idx-1):
            token_strings.append(token_string)
            token_string_index_list.append(token_string_index_list_temp)
            token_string = ""
            token_string_index_list_temp = []
            token_string = "".join([token_string,token_array[token_idx]])
            token_string_index_list_temp.append(token_idx)

        #End of list append remaining string
        if (count_idx == len(idx_list)-1):
            token_strings.append(token_string)
            token_string_index_list.append(token_string_index_list_temp)

    return token_strings, token_string_index_list

#%%
# Adjusts a list of ranges of indices after delete or insert operations
def shift_token_indices_in_list_of_index_lists( idx_list, insert_or_delete_point, direction):
    if direction !=-1 and direction != 1:
        return idx_list
    for idx, idx_sublist in enumerate(idx_list):
        # idx_token points to a token 
        for list_idx, idx_token in enumerate(idx_sublist):
            if idx_token > insert_or_delete_point:
                idx_token = idx_token + direction
                idx_sublist[list_idx] = idx_token

        idx_list[idx] = idx_sublist
            #if direction == -1 : ## für delete
            #try:
                #idx_sublist.remove(insert_or_delete_point)
                #idx_list[idx] = idx_sublist
            #except:
                #return idx_list

    return idx_list
    #%%
    # %%
def find_min(list):
    list2 = list.copy()
    list2.sort()
    return list2[0]
# %%
def find_max(list):
    length = len(list)
    list2 = list.copy()
    list2.sort()
    return list2[length-1]

# %%
# Function to find inverse permutations
def inverse_permutation(arr, size):
 
    # Loop to select Elements one by one
    for i in range(0, size):
 
        # Loop to print position of element
        # where we find an element
        for j in range(0, size):
 
        # checking the element in increasing order
            if (arr[j] == i + 1):
 
                # print position of element where
                # element is in inverse permutation
                print(j + 1, end = " ")
                break

#%%      
def compare_sentence_to_graph(words_sentence_temp, words_graph_temp):
        # Filter out all common tokens in sentence and graph
        for idx_words_sentence_temp, word in enumerate(words_sentence_temp):
          if (word in words_graph_temp):
            #print(word)
            #word_idx_graph = np.where(words_graph_temp==word)[0][0]
            idx_words_graph_temp = words_graph_temp.index(word)
            #print(f"Idx graph: {word_idx_graph}")
            #print(words_graph_temp[word_idx_graph])
            words_graph_temp[idx_words_graph_temp] = "word processed"
            #print(words_sentence_temp[word_idx_sentence])
            words_sentence_temp[idx_words_sentence_temp] = "word processed"

        # Process remaining sentence tokens to match graph tokens
        # Get indices of remaining tokens
        words_graph_temp_np = np.array(words_graph_temp)
        words_sentence_temp_np = np.array(words_sentence_temp)
        remaining_tokens_graph_idx = np.where(words_graph_temp_np!="word processed")[0]
        remaining_tokens_sentence_idx = np.where(words_sentence_temp_np!="word processed")[0]

        return words_sentence_temp, words_graph_temp, remaining_tokens_sentence_idx, remaining_tokens_graph_idx

#%%
def save_sentences(sentences, filename):
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    with open(filename, 'w') as output:
        output.write("\n".join(sentences))
        #output.write("\n")
        output.close
#%%
def save_syntrees(syntax_graphs, filename):
        dirname = os.path.dirname(filename)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        with open(filename, 'wb') as handle:
            pickle.dump(syntax_graphs, handle)
            handle.close