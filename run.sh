#!/bin/bash

source ~/cdaniel/venv_syntrans/bin/activate

if [ $1 == "mlm" ];
then
    rm ./temp/syntrans_masked.py
    jupyter nbconvert --output-dir='./temp' --to script ./syntrans_masked.ipynb
    nohup python3 ./temp/syntrans_masked.py > ./logs/syntrans_masked.log & echo $!
fi

if [ $1 == "syngnn" ];
then
    rm ./temp/syntrans_main.py
    rm ./logs/syntrans_main.log
    jupyter nbconvert --output-dir='./temp' --to script ./syntrans_main.ipynb
    nohup python3 ./temp/syntrans_ner.py > ./logs/syntrans_main.log & echo $!
fi

if [ $1 == "preprocessing_ud" ];
then
    rm ./logs/data_preprocessing_dep_ud.log
    nohup python3 ./data_preprocessing_dep_ud.py > ./logs/data_preprocessing_ud.log & echo $!
fi

if [ $1 == "nertagging" ];
then
    rm ./logs/ner_tagging.log
    nohup python3 ./ner_tagging.py > ./logs/ner_tagging.log & echo $!
fi

