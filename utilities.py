import os


# %%
# Available Bert configurations
saved_models = [
    "bert-base-cased", #0
    "bert-base-uncased", #1
    "shrdlu9/bert-base-cased-ud-NER" # 2 Bert cased trained with dependency tree 
    "shrdlu9/bert-base-cased-ud-NER" # 3 Bert cased trained with dependency tree 
    "./trained_models/ner/bert/dep/ner_bert-base-cased_E9_batches32_LR2e-05_SL96_GN0-0", #4 Bert cased trained with dependency tree data
    "./trained_models/ner/bert/dep/09_21_bert-base-uncased_E9_batches32_LR2e-05_SL96_GN0-0", #5 Bert uncased trained with dependency tree data
    "./trained_models/ner/bert/const/09_27_bert-base-cased_E9_batches32_LR2e-05_SL96_GN0", #6 Bert cased trained with const tree data
    "./trained_models/ner/bert/const/09_28_bert-base-uncased_E9_batches32_LR2e-05_SL96_GN0", #7 Bert uncased trained with const tree data
    "./trained_models/ner/bert/dep/10_23_bert-base-cased_E9_batches32_LR2e-05_SL96_GN0-0_LWFalse", #8 Bert cased trained with balanced dependency tree data
    "./trained_models/ner/bert/dep/10_23_bert-base-uncased_E9_batches32_LR2e-05_SL96_GN0-0_LWFalse", #9 Bert uncased trained with balanced dependency tree data    
]
# %%
label_weights_ud = []
# %%
class Params:
    def __init__(self, use_gnn, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences, max_grad_norm, num_att_heads, num_layers, use_label_weights, use_grammar, trained_epochs, syntree_type):
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
        self.syntree_type = syntree_type
        self.label_weights_clip = 50
        #self.lr_decay = 0.3 # Bert pretraining
        #self.lr_decay_end = 5 # Bert pretraining
        self.lr_decay = 0.2 # Bert pretraining
        self.lr_decay_end = 8 # Bert pretraining
        self.trained_epochs = trained_epochs

    # %%

def configureParameters(parameters):
        # Saved model path
        parameters["saved_model_path"] = parameters["saved_model_path"].astype('string') 
        saved_model_path = parameters["saved_model_path"][0]
        resume_train = parameters["resume_train"][0]

        # Check if saved model path is index to saved_models array
        try:
            if int(saved_model_path) < len(saved_models):
                saved_model_path = saved_models[int(saved_model_path)]
                print(f"Loading model from path {saved_model_path}")
        except:
            print(f"Loading model from path {saved_model_path}")
        # Extract tokenizer from saved model path
        if(len(saved_model_path.split("/")) > 1 and saved_model_path.find("shrdlu9") == -1):
            transformer_name = saved_model_path.split("/")[-1]
            model_config = transformer_name.split("_")
            tokenizer = [item for item in model_config if item.startswith('bert')][0]
            #tokenizer = transformer_name.split("_")[1]
            sequence_length = [int(item.replace("SL", "")) for item in model_config if item.startswith('SL')][0]
            if resume_train == "True:":
                trained_epochs = [int(item.replace("E", "")) for item in model_config if item.startswith('E')][0]
            else:
                trained_epochs = 0
        else:
            if(saved_model_path.find("bert-base-cased") != -1):
                tokenizer = "bert-base-cased"
            elif (saved_model_path.find("bert-base-uncased") != -1):
                tokenizer = "bert-base-uncased"
            else:
                tokenizer = "bert-base-cased"
            # sequence length norm parameter
            if( "seq_len" in parameters):
                sequence_length = int(parameters["seq_len"])
            else:
                sequence_length = 96
            trained_epochs = 0
        
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
        
        # Selected syntree type: 'gold-standard hand annotated' (gold) or automatically generated  (gen)
        syntree_type = parameters["syntree_type"][0]




        return Params(use_gnn, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences, max_grad_norm, num_att_heads, num_layers, use_label_weights, use_grammar, trained_epochs, syntree_type)


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