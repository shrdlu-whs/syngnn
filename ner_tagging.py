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


def CreateNERLabelsFromDataset(file, tagger):

    ud_file = os.path.abspath(file)
    filename = os.path.basename(ud_file)
    print(filename)
    with open(ud_file) as fp:
        lines = fp.readlines()
    
    # Save NER tags
    dirname = os.path.dirname(ud_file)
    dirname = dirname + "/ner/"

    if not os.path.exists(dirname):
        os.makedirs(dirname)

    ner_filepath = os.path.join(dirname, filename)
    filename_ner = ner_filepath.split(".")[0] + "-orig.ner"
    filename_manual_ner = ner_filepath.split(".")[0] + "-manual.ner"
    with open(filename_ner, 'w') as file_ner:
        with open(filename_manual_ner, 'a') as file_manual_ner:
            for idx, sentence in enumerate(lines):
                # Write sentence to file for manual correction
                sentence = Sentence(sentence)
                # predict NER tags
                tagger.predict(sentence, force_token_predictions=True)

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


# Load text files
data_path = "./data/ud/"
# Files in data folder to ignore
skip_files = ["en_gum-ud-dev.txt", "en_gum-ud-test.txt", "en_gum-ud-train.txt"]
files = glob.iglob(data_path + '**/en_*.txt', recursive=True)
files = [f for f in files if all(sf not in f for sf in skip_files)]

mode = "CREATE"

if(mode == "CREATE"):
    # load tagger
    tagger = SequenceTagger.load("flair/ner-english-ontonotes-large")
    print(tagger.label_dictionary)
    for ud_file in files:

        CreateNERLabelsFromDataset(ud_file, tagger)

elif (mode == "CONVERT"):
    for manual_ner_file in files:
        ConvertManuallyCorrectedNERFile(manual_ner_file)

       
    
    