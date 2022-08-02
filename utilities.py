# %%
class Params:
    def __init__(self, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences):
        self.saved_model_path = saved_model_path
        self.train_model = train_model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.task = task
        self.max_grad_norm = 0.0
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

        return Params(saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task, num_threads, num_sentences)
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