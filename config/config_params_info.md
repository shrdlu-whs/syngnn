The configuration files determine the runtime configuration for the models. In particular, you can choose which model and dataset to load, whether to use the SynGNN or BERT only and if you want to train the model or just evaluate the loaded model.<br/>
The available parameters are:
<br/>

- **train_model**: Whether to train the loaded model or just evaluate it on the given task.

    | Argument |                              |
    |----------|------------------------------|
    | True  | Train and evaluate loaded model |
    | False | Evaluate loaded model           |
    

- **epochs**: Number of epochs to train the model for


- **batch_size**: data batch size. Default: 32


- **num_threads**: Maximum number of threads to use


- **learning_rate**: Model learning rate. 


- **saved_model_path**: Expects the path to a BERT model. Accepts any model path available on Huggingface, e.g. 'bert-base-cased' or a path stored locally when training the model. Alternatively, some model paths on Huggingface are predefined in utilities.py and can be loaded by giving their number.

    | Argument | Model                                                 |
    |----------|-------------------------------------------------------|
    | 0        | original bert-base-cased                              |
    | 1        | original bert-base-uncased                            |
    | 2        | bert-base-cased pre-trained on UD data and NER task   |
    | 3        | bert-base-uncased pre-trained on UD data and NER task |



- **data_path**: path where the input data is located. 

- **task**: Model task to evaluate on

    | Argument |                               |
    |----------|-------------------------------|
    | ner      | Named Entity Recognition task |
    | mlm      | Masked Language Modeling task |


- **num_sentences**: Number of sentences to load from each input file. Useful for faster testing. 0 = Load all available sentences.


- **use_gnn**: Whether to use the BERT baseline model or the SynGNN model. If True, num_layers, num_att_heads and syntree_type must also be set.

    | Argument |             |                 
    |----------|-------------|
    | True  | Use SynGNN     |
    | False | Use BERT model |


- **max_grad_norm**: Factor to use for gradient normalization. Default: 0.0


- **num_layers**: Number of SynGNN layers


- **num_att_heads**: Number of attention heads in one SynGNN layer


- **seq_len**: BERT sequence length to use


- **use_weights**: Whether to calculate label weights from training data
    
    | Argument |                               |
    |----------|-------------------------------|
    | True  | Calculate label weights from data|
    | False | Set all labels weights to 1      |


- **use_grammar**: Specify whether to use constituency or dependency grammar
    
    | Argument |                       |                 
    |----------|-----------------------|
    | dep  | Use dependency grammar    |
    | const | Use constituency grammar |


- **syntree_type**: Specify whether gold-standard syntax trees are used (expects -gold.syntree files) or generated syntax trees are used (-gen.syntree files).

    | Argument |                       |                 
    |----------|-----------------------|
    | gold  | Use gold-standard trees  |
    | gen | Use generated trees        |


- **resume_train**: Whether to start the training epoch counter at 0 or infer trained epochs from folder name and resume epoch counter from there. Only works with trained models saved by the syngnn_main.ipynb process. Default: False

    | Argument |                                      |                 
    |----------|--------------------------------------|
    | True  | Infer trained epochs from model folder  |
    | False| Start epochs at 0                        |


