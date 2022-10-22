###########################
# Named Entity Recognition tagging with FLAIR / FLERT
# Automatic tagging of UD dataset for manual correction
###########################

from flair.data import Sentence
from flair.models import SequenceTagger
import os
import glob
import re
import torch
import random
import pickle
import pandas as pd

# Export env vars to limit number of threads to use
num_threads = str(12)
os.environ["OMP_NUM_THREADS"] = num_threads 
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads 
os.environ["VECLIB_MAXIMUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads

# Only use CPU, hide GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Limit no. of threads used by Pytorch
torch.set_num_threads = int(num_threads)

PID = os.getpid()
PGID = os.getpgid(PID)
print(f"PID: {PID}, PGID: {PGID}")

# Only use CPU, hide GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def BalanceNERFile(ner_file, discard_sentences_ratio = 0.25, discard_ne_ratio = 0.45):

    # Extract syntree filepaths
    syntree_file_gold = ner_file.replace("ner/","").replace("-orig.ner","-gold.syntree")
    syntree_file_gold_balanced = ner_file.replace("ner/","").replace("-orig.ner","-gold-balanced.syntree")
    print(syntree_file_gold)
    print(syntree_file_gold_balanced)

    # Save number of written senteces and NE tags
    written_sentences = 0
    written_ne_tags = 0

    ner_file = os.path.abspath(ner_file)
    filename = os.path.basename(ner_file)
    print(filename)
    ner_file_balanced = ner_file.replace("orig", "orig-balanced")
    print(ner_file_balanced)
    with open(ner_file) as fp:
        sentences = fp.readlines()
    with open(syntree_file_gold, 'rb') as fp:
        syntrees = pd.read_pickle(fp)
    syntree_balanced_idx_list = []

    with open(ner_file_balanced, 'w') as file_ner:

        for idx, sentence in enumerate(sentences):
            #tokens_tags = line.split('\t')

            #print(tokens_tags)
            #tokens_sentence = tokens_tags[0]
            tokens_tags = [x.split() for x in sentence.split("\t")]

            write_sentence = True

            ne_tags = 0
            random_number = random.uniform(0,1)
            # Check if sentence contains NE tags
            for token_tag_pair in tokens_tags:
                if token_tag_pair == []:
                    continue
                ne_tag = token_tag_pair[1]
                
                if ne_tag != "O":
                    # Count only single tokens and beginnings of words to get tag count
                    if ne_tag.find("S-") != -1:
                        ne_tags = ne_tags+1
                    if ne_tag.find("B-") != -1:
                        ne_tags = ne_tags+1
            # Sentences contains no NE: discard according to discard sentences ratio
            if ne_tags == 0 and random_number < discard_sentences_ratio:
                write_sentence = False
            # # Sentence contains NE: discard according to discard NE ratio
            if ne_tags > 0 and random_number < discard_ne_ratio:
                write_sentence = False

            if write_sentence == True:
                written_sentences = written_sentences+1
                written_ne_tags = written_ne_tags+ne_tags
                file_ner.write(sentence)
                syntree_balanced_idx_list.append(idx)

            
    balanced_syntrees = [x for idx, x in enumerate(syntrees) if idx in syntree_balanced_idx_list]

    # Save list of balanced syntrees
    dirname = os.path.dirname(syntree_file_gold_balanced)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(syntree_file_gold_balanced, 'wb') as handle:
        pickle.dump(balanced_syntrees, handle)    

    return written_sentences, written_ne_tags


def ConvertManuallyCorrectedNERFile(manual_ner_file):

    manual_ner_file = os.path.abspath(manual_ner_file)
    filename = os.path.basename(manual_ner_file)
    print(filename)
    ner_file = manual_ner_file.replace("manual", "corr")
    with open(manual_ner_file) as fp:
        lines = fp.readlines()
    

    with open(ner_file, 'w') as file_ner:

        sentence = ""
        token_ner_list = []

        for line in lines:

            if(line.startswith("Sentence: ")):

                # Previous Sentence NER labels are available: write sentence to file
                if(len(token_ner_list) > 0):
                    for idx, token in enumerate(sentence):
                        file_ner.write(token + " " + token_ner_list[idx] + "\n")
                    # Reset NER label list
                    token_ner_list = []

                # Extract sentence only
                sentence = line.replace("Sentence: ", "", 1).split(" ")
                # Initialize list of NER labels with O [= token is no NE]
                token_ner_list = ["O"] * len(sentence)

            # Line is NER label: Extract token index and NER label
            elif(line.startswith("Token[")):
                regex = re.search("^Token\[(\d+)\]: \"(.+)\" → (.+) \(", line)
                # Extract index of token with NER label
                token_idx = int(regex.group(1))

                # Extract NER label
                ner_label = regex.group(3)
                token_ner_list[token_idx] = ner_label


def CreateNERLabelsFromDataset(file, tagger, balance_dataset = False, discard_sentences_ratio = 0.25, discard_ne_ratio = 0.45):

    ud_file = os.path.abspath(file)
    filename = os.path.basename(ud_file)
    print(filename)
    with open(ud_file) as fp:
        lines = fp.readlines()
        print(f"Number of sentences: {len(lines)}")
    
    # Save NER tags
    dirname = os.path.dirname(ud_file)
    dirname = dirname + "/ner/"

    # Save number of written senteces and NE tags
    written_sentences = 0
    written_ne_tags = 0

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    ner_filepath = os.path.join(dirname, filename)
    if balance_dataset == True:
        filename_ner = ner_filepath.split(".")[0] + "-orig-balanced.ner"
        filename_manual_ner = ner_filepath.split(".")[0] + "-manual-balanced.ner"
    else:
        filename_ner = ner_filepath.split(".")[0] + "-orig.ner"
        filename_manual_ner = ner_filepath.split(".")[0] + "-manual.ner"
    with open(filename_ner, 'w') as file_ner:
        with open(filename_manual_ner, 'a') as file_manual_ner:
            
            for idx, sentence in enumerate(lines):
                # Write sentence to file for manual correction
                sentence = Sentence(sentence)
                # predict NER tags
                tagger.predict(sentence, force_token_predictions=True)

                write_sentence = True
                if balance_dataset == True:

                    ne_tags = 0
                    random_number = random.uniform(0,1)
                    # Check if sentence contains NE tags
                    for token in sentence.tokens:
                        if token.get_label().value != "0":
                            ne_tags = ne_tags+1
                    # Sentences contains no NE: discard according to discard sentences ratio
                    if ne_tags == 0 and random_number < discard_sentences_ratio:
                        write_sentence = False
                    # Sentence contains NE: discard according to discard NE ratio
                    if ne_tags > 0 and random_number < discard_ne_ratio:
                        write_sentence = False

                if write_sentence == True:
                    written_sentences = written_sentences+1
                    written_ne_tags = written_ne_tags+ne_tags
                    file_manual_ner.write("Sentence: ")
                    # Write BIOES NER tags to file
                    for token in sentence.tokens:
                        file_ner.write(token.text + " " + token.get_label().value+"\t")
                        file_manual_ner.write(token.text + " ")
                    file_ner.write("\n")
                    file_manual_ner.write("\n")

                    # Write identified NER tags to file for manual correction
                    for label in sentence.get_labels('ner'):
                        file_manual_ner.write(str(label) +"\n")
                
            return written_sentences, written_ne_tags


# Load text files
data_path = "./data/ud/"
# Files in data folder to ignore
skip_files = ["./data/ud/testdata/*"]
#files = glob.iglob(data_path + '**/en_gum-ud-test-bert-base-cased.txt', recursive=True)
files = glob.iglob(data_path+"**/ner/*-test-*-cased-orig.ner")
files = [f for f in files if all(sf not in f for sf in skip_files)]

mode = "BALANCE"

if(mode == "CREATE"):
    # load tagger
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    print(tagger.label_dictionary)
    for ud_file in files:

        written_sentences, written_ne_tags = CreateNERLabelsFromDataset(ud_file, tagger, balance_dataset=True)



elif (mode == "CONVERT"):
    for manual_ner_file in files:
        ConvertManuallyCorrectedNERFile(manual_ner_file)

elif (mode == "BALANCE"):
    discard_sentences_ratio = 0.25
    discard_ne_ratio = 0.45
    #discard_sentences_ratio = 0.0
    #discard_ne_ratio = 0.0
    total_sentences = 0
    total_ne_tags = 0
    for ner_file in files:
        print(ner_file)
        written_sentences, written_ne_tags = BalanceNERFile(ner_file, discard_sentences_ratio=discard_sentences_ratio, discard_ne_ratio=discard_ne_ratio)
        print(f"Sentences: {written_sentences}")
        print(f"NE tags: {written_ne_tags}")

        total_sentences = total_sentences+written_sentences
        total_ne_tags = total_ne_tags+written_ne_tags

    print(f"Total Sentences: {total_sentences}")
    print(f"Total NE tags: {total_ne_tags}")
    
    