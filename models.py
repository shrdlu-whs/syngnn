# %%
from ast import Try
from transformers import BertTokenizer, BertForMaskedLM, BertForTokenClassification, BertConfig, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F
#Import SummaryWriter for Tensorboard logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import (DataLoader, TensorDataset)
# Load Pytorch Geometric
from torch_geometric.data import Data
import torch_geometric.data as tg_data
import torch_geometric.loader as tg_loader
import torch_geometric.utils as tg_utils
import torch_geometric.nn as tg_nn
import copy
import random
import numpy as np
import os
import torch.jit as jit


# %%
class BertForNer(BertForTokenClassification):
    """
    Adapted from Huggingface BertForTokenClassification
    """

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, label_ids=None,valid_ids=None,attention_mask_label=None):

        # Calculate new embeddings
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape

        # Initialize valid output
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32)
        # Calculate new sequence output: ignore non-valid tokens, e.g. subtokens of words
        for batch_idx in range(batch_size):
            valid_idx = -1
            for token_idx in range(max_len):
                    if valid_ids[batch_idx][token_idx].item() == 1:
                        valid_idx += 1
                        valid_output[batch_idx][valid_idx] = sequence_output[batch_idx][token_idx]
        sequence_output = self.dropout(valid_output)
        logits = self.classifier(sequence_output)

        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # Ignore padding tokens in loss calculation
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
            return loss, logits
        else:
            return logits
# %%
def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    elif activation == "sigmoid":
        return F.sigmoid
    else:
        raise RuntimeError("activation should be relu/gelu/sigmoid, not {}".format(activation))

class SynGNNLayer(torch.nn.Module):
    """
    SynGNN Pytorch module
    based on Pytorch TransformerEncoderLayer implementing the architecture in paper “Attention Is All You Need”. 
    
    """
    def __init__(self, dim_in, dim_out, num_att_heads, dim_edge_attrs=None, dropout=0.1, activation="gelu", dim_feedforward=2048):
        r"""
        Args:
            param dim_in: input dimension
            param dim_hdn: hidden nodes dimension
            param dim_out: output dimension
        """
        super(SynGNNLayer, self).__init__()
        # Graph attention sublayer
        self.graph_attn = tg_nn.GATv2Conv(in_channels=dim_in, out_channels=dim_in, heads=num_att_heads, edge_dim =dim_edge_attrs, concat=False)
        self.linear1 = tg_nn.Linear(dim_in, dim_feedforward)
        self.linear2 = tg_nn.Linear(dim_feedforward, dim_in)
        #self.linear_classifier = tg_nn.Linear(dim_in, dim_out)

        self.norm0 = tg_nn.LayerNorm(dim_in)
        self.norm1 = tg_nn.LayerNorm(dim_in)
        self.norm2 = tg_nn.LayerNorm(dim_in)
        #self.norm3 = tg_nn.LayerNorm(dim_out)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = get_activation_fn(activation)
    
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(SynGNNLayer, self).__setstate__(state)

    
    def forward(self, x, edge_index, edge_attr, batch):
        r"""Pass the input through the encoder layer.
        Args:
            x: node features
            edge_index: graph edges
            batch: current batch
        """
        #print(f"Input: {x.size()}")
        # Graph attention sublayer
        src = self.norm0(x)
        #print(f"After norm: {src.size()}")
        src2, att = self.graph_attn(src, edge_index, edge_attr, return_attention_weights = True)

        #print(f"Graph Attention Output src2: {src2.size()}")
        src = src + self.dropout1(src2)
        #print(f" After residual connection: {src.size()}")
        src = self.norm1(src)

        # Feed-Forward-Network sublayer
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))

        #print(f" After linear layer 2: {src2.size()}")
        src = src + self.dropout2(src2)
        #print(f" After residual connection: {src.size()}")
        src = self.norm2(src)
        #src = self.linear_classifier(src)
        #src = self.norm3(src)
        #print(f" After linear output layer: {src.size()}")
        return src, att

# %%
class HighwayFcNet(nn.Module):
	"""
		A more robust fully connected network
		return: H*T + (1-T)x
	"""
	def __init__(self, input_size, activation_type='sigmoid',gate_activation='sigmoid',bias=-1.0): #activation_type is a string containing the name of the activation
		"""
        Highway network
		"""
		super(HighwayFcNet,self).__init__()
		self.activation = get_activation_fn(activation_type) #H func
		#self.gate_activation = get_activation_fn(gate_activation)#T func
		self.plain = nn.Linear(input_size,input_size)
		nn.init.xavier_uniform(self.plain.weight)
		self.gate = nn.Linear(input_size,input_size)
		self.gate.bias.data.fill_(bias)

    #def __setstate__(self, state):
        #if 'activation' not in state:
        #    state['activation'] = get_activation_fn('gelu')
        #    state['gate_activation'] = get_activation_fn('sigmoid')
        #super(HighwayFcNet, self).__setstate__(state)

	def forward(self,bert_out, syngnn_out):
		g_out = self.activation(self.plain(bert_out))

		#t_out = self.gate_activation(self.gate(x))

		return torch.add(torch.mul(g_out,bert_out),torch.mul((1.0-g_out),syngnn_out))
# %%
class SynGNN(torch.nn.Module):
    r"""TransformerEncoder is a stack of N encoder layers. 
    Based on Huggingface Pytorch implementation
    Args:
        encoder_layer: an instance of the TransformerEncoderLayer() class (required).
        num_layers: the number of sub-encoder-layers in the encoder (required).
        norm: the layer normalization component (optional).
    Examples::
        >>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
        >>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
        >>> src = torch.rand(10, 32, 512)
        >>> out = transformer_encoder(src)
    """
    __constants__ = ['norm']

    def __init__(self, num_node_features, num_labels, num_att_heads, num_edge_attrs, num_layers, norm=None):
        
        super(SynGNN, self).__init__()
        syngnn_layer = SynGNNLayer(num_node_features, num_labels, num_att_heads, num_edge_attrs)
        self.layers = _get_clones(syngnn_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    
    def forward(self, x, edge_index, edge_attr, batch):
        r"""Pass the input through the encoder layers in turn.
        Args:
            x: node features
            edge_index: graph edges
            edge_attr: graph edge attributes
            batch: current batch
        """
        output = x
        attns = []

        for layer in self.layers:
            output, attn = layer(x, edge_index, edge_attr, batch)
            attns.append(attn)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns
    
    def add_bert_embeddings_to_graph(self, ptg_graph, pt_embeddings, sentence_graph_idx_map, input_ids):
        graph_ids = []
        for sent_token_idx, pt_embedding in enumerate(pt_embeddings):

            if sent_token_idx in sentence_graph_idx_map:
                # Look up corresponding index in graph for current token embedding
                graph_token_idx = sentence_graph_idx_map[sent_token_idx]
                #print(ptg_graph.x[graph_idx])
                #ptg_graph.x.resize(ptg_graph.num_nodes, embedding.size())
                ptg_graph.x[graph_token_idx] = pt_embedding
                graph_ids.append(input_ids[sent_token_idx])

        """print("Tokens graph:")
        print(tokenizer.convert_ids_to_tokens(graph_ids))
        print("Sentence_graph_idx_map:")
        print(sentence_graph_idx_map)"""
        return ptg_graph


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])        
# %%
class SynBertForNer(nn.Module):
    def __init__(self, bert_config, num_node_features, num_labels, num_edge_attrs, num_att_heads, num_layers):

        super(SynBertForNer, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel(bert_config)
        self.syngnn = SynGNN(num_node_features, num_labels, num_att_heads, num_edge_attrs, num_layers = num_layers)
        self.highway = HighwayFcNet(768)
        self.linear_classifier = tg_nn.Linear(768, num_labels)
        self.norm = tg_nn.LayerNorm(num_labels)

    def forward(self, input_ids, syntax_graphs, sentence_graph_idx_maps, token_type_ids=None, attention_mask=None, label_ids=None, label_weights=None, valid_ids=None,attention_mask_label=None):

        # Calculate Bert embeddings
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]

        batch_size,seq_len,feat_dim = sequence_output.shape

        '''# Calculate final sequence output: ignore non-valid tokens, e.g. subtokens of words
        valid_output = torch.zeros(batch_size,seq_len,feat_dim,dtype=torch.float32)
        for batch_idx in range(batch_size):
            valid_idx = -1
            for token_idx in range(seq_len):
                # Only add embedding to output if valid_id mask is 1
                if valid_ids[batch_idx][token_idx].item() == 1:
                    valid_idx += 1
                    valid_output[batch_idx][valid_idx] = sequence_output[batch_idx][token_idx]'''

        # Add Bert embeddings as node features to syntax graph nodes
        graphs_with_embeddings = []

        for batch_idx in range(batch_size):
            graphs_with_embeddings.append(self.syngnn.add_bert_embeddings_to_graph(syntax_graphs[batch_idx], sequence_output[batch_idx],sentence_graph_idx_maps[batch_idx], input_ids[batch_idx]))
       
        pyg_data_batch = tg_data.Batch.from_data_list(graphs_with_embeddings)
        pyg_data_batch.to(torch.device('cpu'))
        
        # Calculate syngnn output
        syngnn_output, attn = self.syngnn(torch.as_tensor(pyg_data_batch.x, dtype=torch.float), pyg_data_batch.edge_index, pyg_data_batch.edge_attr, pyg_data_batch.batch)

        syngnn_in_bert_format = torch.zeros([batch_size,96, 768], dtype=torch.float)
        sentence_position = 0
        sentence_length_ctr = 0
        for batch_idx in range(batch_size):

            sentence_length = syntax_graphs[batch_idx].x.shape[0]-1
            for token_idx in range(sentence_length):
                syngnn_in_bert_format[batch_idx,token_idx,:] = syngnn_output[sentence_position+token_idx,:].detach()
                sentence_length_ctr = sentence_length_ctr+1
            sentence_position = sentence_position+sentence_length


        # Process through highway gate
        highway_output = self.highway(sequence_output, syngnn_in_bert_format)
        # 

        #print(syngnn_in_bert_format[3][0])
        #print(syngnn_in_bert_format[3][85])
        #print(syngnn_in_bert_format[0][0].subtract(syngnn_output[0]))
        #print(np.array(syngnn_in_bert_format).shape)

        # Convert valid_ids mask to Syngnn format: num_tokens*embeddings_size
        # Trim Bert valid ids mask to graph token labels length
        """      valid_ids_mask_syngnn = []
        for sentence_idx, id_mask in enumerate(valid_ids):
            # Find SEP token id in sentence and get index
            sep_idx = label_ids.tolist()[sentence_idx].index(78)
            # Find CLS token id in sentence and get index
            cls_idx = label_ids.tolist()[sentence_idx].index(77)
            valid_ids_mask_temp = id_mask.tolist()
            # Ignore SEP token
            valid_ids_mask_temp[sep_idx] = 0
            # Ignore CLS token
            valid_ids_mask_temp[cls_idx] = 0
            valid_ids_mask_temp = valid_ids_mask_temp[0:sep_idx]
            valid_ids_mask_syngnn.extend(valid_ids_mask_temp)
        valid_ids_mask_syngnn = torch.tensor(valid_ids_mask_syngnn)
        #print(valid_ids_mask_syngnn.size())


        # Calculate final sequence output: ignore non-valid tokens, e.g. subtokens of words
        valid_output = torch.zeros(num_tokens, embedding_dim,dtype=torch.float32)
        #print(valid_output.size())
        valid_idx = -1
        for idx, mask in enumerate(valid_ids_mask_syngnn):
            # Only add embedding to output if valid_id mask is 1
            if mask.item() == 1:
                    valid_idx += 1
                    valid_output[valid_idx] = syngnn_output[idx]"""
        # Calculate classifier output
        #valid_output = self.dropout(valid_output)
        logits = self.linear_classifier(highway_output)
        
        if label_ids is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            # Ignore padding tokens in loss calculation
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = label_ids.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), label_ids.view(-1))
            return loss, logits
        else:
            return logits

        """# Calculate loss if true labels given
        if label_ids is not None:
            # Trim Bert label attention mask to graph token labels length
            token_mask_labels = []
            for sentence_idx, label_mask in enumerate(attention_mask_label):
                # Find SEP token id in sentence and get index
                sep_idx = label_ids.tolist()[sentence_idx].index(78)
                # Find CLS token id in sentence and get index
                cls_idx = label_ids.tolist()[sentence_idx].index(77)
                token_mask_labels_temp = label_mask.tolist()
                # Ignore SEP token
                token_mask_labels_temp[sep_idx] = 0
                # Ignore CLS token
                token_mask_labels_temp[cls_idx] = 0
                token_mask_labels.extend(token_mask_labels_temp)
            token_mask_labels = torch.tensor(token_mask_labels)



            logits_view = logits.view(-1, self.num_labels)
            labels_view = label_ids.view(-1)

            #print(label_weights[0])
            # Loss function: do not count labels with index 1, that is tokens labelled with O (=ignore)
            loss_fct = nn.CrossEntropyLoss(ignore_index=0, weight=label_weights[0])
            # Trim Bert labels to contain only graph token labels, ignore padding
            if attention_mask_label is not None:
                active_labels_mask = token_mask_labels.view(-1) == 1
                active_labels = labels_view[active_labels_mask]
                #try:
                loss = loss_fct(logits_view, active_labels)
                #print(loss)
                '''except:
                print("Logits:")
                print(logits.size())
                print(logits_view.size())
                print("Labels:")
                print(label_ids.size())
                print(active_labels.size())
                print("Input Ids")
                print(input_ids)
                print(input_ids.size())'''

            else:
                loss = loss_fct(logits_view, labels_view)
            return loss, logits
        else:
            return logits"""