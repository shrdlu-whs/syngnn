#!/bin/bash

source ~/cdaniel/venv_syntrans/bin/activate
if [ $1 == "mlm" ];
then
    rm ./temp/syntrans_masked.py
    jupyter nbconvert --output-dir='./temp' --to script ./syntrans_masked.ipynb
    nohup python ./temp/syntrans_masked.py > ./logs/syntrans_masked.log & echo $!
fi

if [ $1 == "ner" ];
then
    rm ./temp/syntrans_ner.py
    rm ./syntrans_ner.log
    jupyter nbconvert jupyter nbconvert --output-dir='./temp' --to script ./syntrans_ner.ipynb
    nohup python ./temp/syntrans_ner.py > ./logs/syntrans_ner.log & echo $!
fi

if [ $1 == "preprocessing" ];
then
    rm ./data_preprocessing.log
    nohup python ./data_preprocessing.py > ./logs/data_preprocessing.log & echo $!
fi

if [ $1 == "nertagging" ];
then
    rm ./ner_tagging.log
    nohup python ./ner_tagging.py > ./logs/ner_tagging.log & echo $!
fi

