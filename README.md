## Description
This is the accompanying code for the thesis 'Comparison of dependency and constituency grammar with syntax-enhanced transformers'. It consists of a SynGNN: a BERT model enhanced with a GNN able to process linguistic syntax graphs.

## Installation
To install, create a new Python environment using Python 3.8.10:
apt install python3.8-venv
python3 -m venv ~/venv_syngnn

Next, install all requirements from the requirements file:
pip3 install -r requirements.txt
To be able to generate your own syntax graphs, also install the spaCy model for the English language:
python -m spacy download en_core_web_lg

## Data
There were two datasets used for the experiments:
1) Universal Dependencies (UD) dataset with dependency graphs: https://universaldependencies.org/en/index.html
2) International Corpus of English GB (ICE-GB) with constituency graphs: https://www.ucl.ac.uk/english-usage/projects/ice-gb/

The UD dataset is freely available for download, but for some of the datasets (e.g. UD_English-ESL) the text and the syntax graphs are distributed separately for copyright reasons. The ICE-GB dataset is available for restricted use for a fee.
Because of these restrictions, this repository only includes a sample from the UD dataset for testing purposes. The full datasets will have to be acquired separately.

## Data Preprocessing
The UD dataset is given as *.conllu files containing the syntax graphs. To use the data for model training, the text and the syntax graphs will have to be extracted and stored as text and Pytorch Geometric graph structures.
To do this, you can run data_preprocessing_dep_ud.py with:
./run.sh preprocessing_ud

By default, this will process all conllu files available in the data_sample/original/ud/ folder and produce the following files from each file:
* a text file containing all sentences
* a *-gold.syntree file containing the Pytorch Geometric graphs and the token mapping from sentence to graph

Likewise, the ICE-GB *.tre files can be processed by running:
./run.sh preprocessing_ice

It is also possible to generate syntax graphs automatically from text files. 

To successfully run the Named Entity Recognition task, we also need NE tags. These can be generated from the text files with:
./run.sh nertagging
This will produce the following files:
* a *-orig.ner file containing the generated NE tags
* a *-manual.ner file containing the NE tags along with the sentence for a more-human readable format. This is meant to facilitate easy correction of wrong NE tags. These files can be converted to the model-compatible format by running ner_tagging.py in CONVERT mode.

## Usage
The main runfile for the models is "syngnn_main.ipynb". You can run the Notebook directly or use the headless server mode with:
./run.sh syngnn
This will convert the Notebook to a Python file and save its output to a log file in logs/syngnn_main.log

The runtime configuration is given by the config files "develop.csv" and "prod.csv". For more information on the config parameters available, see the [config_params_info.md](https://github.com/shrdlu-whs/syngnn/blob/master/config_params_info.md) file.
The results of the model run for each epoch are stored in the logs folder.
The data collected by Tensorboard for the model run is stored in the runs folder.
If the production configuration is active, the model is also saved in trained_models. This is currently only available, when training a baseline BERT model, SynGNN models cannot be stored.

The trained_models folder already includes a pre-trained BERT model that was trained on the full UD data.



