# %%
class Params:
    def __init__(self, saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task):
        self.saved_model_path = saved_model_path
        self.train_model = train_model
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.data_path = data_path
        self.sequence_length = sequence_length
        self.task = task

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
            sequence_length = 136
        
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

        # Selected task: Masked Language Model (mlm) or Named Entity Recognition (ner)

        task = parameters["task"][0]

        return Params(saved_model_path, tokenizer, data_path, train_model, epochs, learning_rate, batch_size, sequence_length, task)
    