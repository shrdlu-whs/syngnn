import os
# %%
class Params:
    def __init__(self, use_gnn, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences, max_grad_norm):
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

    # %%

def configureParameters(parameters):
        # Saved model path
        parameters["saved_model_path"] = parameters["saved_model_path"].astype('string') 
        saved_model_path = parameters["saved_model_path"][0]
        # Extract tokenizer from saved model path
        if(len(saved_model_path.split("/")) > 1 ):
            transformer_name = saved_model_path.split("/")[-1]
            tokenizer = transformer_name.split("_")[0]
            sequence_length = int(transformer_name.split("_")[-1].replace("SL",""))
        else:
            tokenizer = saved_model_path
            #sequence_length = 136
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
            print(parameters["num_sentences"])
            num_sentences = int(parameters["num_sentences"])
        else:
            num_sentences = 0
        
         # Number of sentences to process in each file
        if( "max_grad_norm" in parameters):
            print(parameters["num_sentences"])
            max_grad_norm = float(parameters["max_grad_norm"])
        else:
            max_grad_norm = None

        return Params(use_gnn, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences, max_grad_norm)
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
def inversePermutation(arr, size):
 
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

# %%
def createNumberedDir(dirname):
    """
    Create folder with specified name.
    If folder exists and is not empry: create folder with running idx attached
    returns: created folder name
    """
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