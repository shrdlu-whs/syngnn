import os


# %%
# Available Bert configurations
saved_models = [
    "bert-base-cased", #0
    "bert-base-uncased", #1
    "./trained_models/ner/bert/ner_bert-base-cased_E9_batches32_LR2e-05_SL96_GN0-0_0", #2
    "./trained_models/ner/bert/ner_bert-base-cased_E0_batches2_LR2e-05_SL96_GN0-0_0", #3
    "./trained_models/ner/bert/09_21_bert-base-uncased_E9_batches32_LR2e-05_SL96_GN0-0_0", #4
    "./trained_models/mlm/bert/09_22_bert-base-cased_E5_batches32_LR2e-05_SL96_GN0-0_1", #5
    "./trained_models/ner/bert/09_27_bert-base-cased_E9_batches32_LR2e-05_SL96_GN0-0_0" #6 Bert trained with const tree data
]
# %%
label_weights_ud = []
# %%
class Params:
    def __init__(self, use_gnn, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences, max_grad_norm, num_att_heads, num_layers, use_label_weights, use_grammar):
        self.use_gnn = use_gnn
        self.saved_model_path = saved_model_path
        self.train_model = train_model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.task = task
        self.max_grad_norm = max_grad_norm
        self.num_threads = num_threads
        self.num_sentences = num_sentences
        self.num_att_heads = num_att_heads
        self.num_layers = num_layers
        self.use_label_weights = use_label_weights
        self.use_grammar = use_grammar
        self.label_weights_clip = 50
        self.lr_decay = 0.3
        self.lr_decay_end = 5

    # %%

def configureParameters(parameters):
        # Saved model path
        parameters["saved_model_path"] = parameters["saved_model_path"].astype('string') 
        saved_model_path = parameters["saved_model_path"][0]

        # Check if saved model path is index to saved_models array
        try:
            if int(saved_model_path) < len(saved_models):
                saved_model_path = saved_models[int(saved_model_path)]
                print(f"Loading model from path {saved_model_path}")
        except:
            print(f"Loading model from path {saved_model_path}")
        # Extract tokenizer from saved model path
        if(len(saved_model_path.split("/")) > 1 ):
            transformer_name = saved_model_path.split("/")[-1]
            model_config = transformer_name.split("_")
            tokenizer = [item for item in model_config if item.startswith('bert')][0]
            #tokenizer = transformer_name.split("_")[1]
            sequence_length = [int(item.replace("SL", "")) for item in model_config if item.startswith('SL')][0]
            #sequence_length = int(transformer_name.split("_")[5].replace("SL",""))
        else:
            tokenizer = saved_model_path
            # sequence length norm parameter
            if( "seq_len" in parameters):
                sequence_length = int(parameters["seq_len"])
            else:
                sequence_length = 96
        
        # Data path
        data_path = parameters["data_path"][0]

        # Train model
        if(parameters["train_model"][0] == "True" or str(parameters["train_model"][0]) == "1"):
            train_model = True
        else:
            train_model = False
        
        # Use GNN
        if(parameters["use_gnn"][0] == "True" or str(parameters["use_gnn"][0]) == "1"):
            use_gnn = True
        else:
            use_gnn = False
        
        # Number of learning epochs
        epochs = int(parameters["epochs"])
        learning_rate = float(parameters["learning_rate"])
        batch_size = int(parameters["batch_size"])
        num_threads = int(parameters["num_threads"])

        # Selected task: Masked Language Model (mlm) or Named Entity Recognition (ner)
        task = parameters["task"][0]

        # Number of sentences to process in each file
        if( "num_sentences" in parameters):
            num_sentences = int(parameters["num_sentences"])
        else:
            num_sentences = 0
        
         # Max grad norm parameter
        if( "max_grad_norm" in parameters):
            max_grad_norm = float(parameters["max_grad_norm"])
        else:
            max_grad_norm = None
        
        # Num attention heads in GAT layer
        if( "num_att_heads" in parameters):
            num_att_heads = int(parameters["num_att_heads"])
        else:
            num_att_heads = 1
        
        # Num transformer layers
        if( "num_layers" in parameters):
            num_layers = int(parameters["num_layers"])
        else:
            num_layers = 2

        # Use weights
        if(parameters["use_weights"][0] == "True" or str(parameters["use_weights"][0]) == "1"):
            use_label_weights = True
        else:
            use_label_weights = False

        # Use grammar syntax graphs
        if(parameters["use_grammar"][0] == "dep" or str(parameters["use_grammar"][0]) == "const"):
            use_grammar = parameters["use_grammar"][0]
        else:
            print("Select one of 'dep' for dependency style syntax trees or 'const' for constituency style syntax trees")
            exit()




        return Params(use_gnn, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences, max_grad_norm, num_att_heads, num_layers, use_label_weights, use_grammar)


# %%
def createNumberedDir(dirname):
    """
    Create folder with specified name.
    If folder exists and is not empry: create folder with running idx attached
    returns: created folder name
    """
    # Remove ending slash from dir if given
    if dirname[-1] == "/":
        dirname = dirname[:-1]
    log_idx = 0
    while(os.path.exists(dirname+f"_{log_idx}")):
        # Check is dir is not empty
        if(os.listdir(dirname+f"_{log_idx}") and log_idx <30):
            log_idx = log_idx+1
        else:
            break
    dirname = dirname+f"_{log_idx}/"
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    return dirname