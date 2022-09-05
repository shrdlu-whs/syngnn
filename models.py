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
class SynGNNLayer(nn.Module):
    """
    SynGNN Pytorch module
    based on Pytorch TransformerEncoderLayer implementing the architecture in paper “Attention Is All You Need”. 
    
    """
    def __init__(self, dim_in, dim_out, num_att_heads, dim_hdn=2048, dim_edge_attrs=None, dropout=0.1, activation="relu"):
        r"""
        Args:
            param dim_in: input dimension
            param dim_hdn: hidden nodes dimension
            param dim_out: output dimension
        """
        super(SynGNNLayer, self).__init__()
        # Graph attention sublayer
        self.graph_attn = tg_nn.GATv2Conv(in_channels=dim_in, out_channels=dim_in, heads=num_att_heads, edge_dim =dim_edge_attrs, concat=False)
        self.linear1 = tg_nn.Linear(dim_in, dim_hdn)
        self.linear2 = tg_nn.Linear(dim_hdn, dim_in)
        self.linear_classifier = tg_nn.Linear(dim_in, dim_out)

        self.norm0 = tg_nn.LayerNorm(dim_in)
        self.norm1 = tg_nn.LayerNorm(dim_in)
        self.norm2 = tg_nn.LayerNorm(dim_in)
        self.norm3 = tg_nn.LayerNorm(dim_out)
        self.dropout0 = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        def _get_activation_fn(activation):
            if activation == "relu":
                return F.relu
            elif activation == "gelu":
                return F.gelu
            else:
                raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

        self.activation = _get_activation_fn(activation)
    
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

        #print(f"Graph Att Output src2: {src2.size()}")
        src = src + self.dropout1(src2)
        #print(f" After resdual connection: {src.size()}")
        src = self.norm1(src)

        # Feed-Forward-Network sublayer
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))

        #print(f" After linear layer 2: {src2.size()}")
        src = src + self.dropout2(src2)
        #print(f" After residual connection: {src.size()}")
        src = self.norm2(src)
        src = self.linear_classifier(src)
        src = self.norm3(src)
        #print(f" After linear output layer: {src.size()}")
        return src, att

# %%
class SynGNN(nn.Module):
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

    def __init__(self, encoder_layer, num_layers, norm=None):
        super(SynGNN, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
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
        #attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns
    
    def add_bert_embeddings_to_graph(self, ptg_graph, pt_embeddings, sentence_graph_idx_map, input_ids):
        graph_ids = []
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        for sent_token_idx, pt_embedding in enumerate(pt_embeddings):
            embedding = pt_embedding.clone().detach()

            if sent_token_idx in sentence_graph_idx_map:
                # Look up corresponding index in graph for current token embedding
                graph_token_idx = sentence_graph_idx_map[sent_token_idx]
                #print(ptg_graph.x[graph_idx])
                #ptg_graph.x.resize(ptg_graph.num_nodes, embedding.size())
                ptg_graph.x[graph_token_idx] = embedding
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
    def __init__(self, bert_config, num_node_features, num_labels, num_edge_attrs, num_att_heads, num_layers, device_str=None):

        super(SynBertForNer, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel(bert_config)
        self.gnn_layer = SynGNNLayer(dim_in=num_node_features, dim_out=num_labels, dim_edge_attrs=num_edge_attrs,num_att_heads=num_att_heads)
        self.syngnn = SynGNN(self.gnn_layer, num_layers = num_layers)

    def forward(self, input_ids, syntax_graphs, sentence_graph_idx_maps, token_type_ids=None, attention_mask=None, label_ids=None,valid_ids=None,attention_mask_label=None):
        #print("Input Ids")
        #print(input_ids)

        # Calculate Bert embeddings
        sequence_output = self.bert(input_ids, token_type_ids, attention_mask,head_mask=None)[0]
        batch_size,max_len,feat_dim = sequence_output.shape

        # Calculate final sequence output: ignore non-valid tokens, e.g. subtokens of words
        valid_output = torch.zeros(batch_size,max_len,feat_dim,dtype=torch.float32)
        for batch_idx in range(batch_size):
            valid_idx = -1
            for token_idx in range(max_len):
                # Only add embedding to output if valid_id mask is 1
                if valid_ids[batch_idx][token_idx].item() == 1:
                    valid_idx += 1
                    valid_output[batch_idx][valid_idx] = sequence_output[batch_idx][token_idx]

        # Add Bert embeddings as node features to syntax graph nodes
        graphs_with_embeddings = []

        for batch_idx in range(batch_size):
            graphs_with_embeddings.append(self.syngnn.add_bert_embeddings_to_graph(syntax_graphs[batch_idx], valid_output[batch_idx],sentence_graph_idx_maps[batch_idx], input_ids[batch_idx]))
       

        # Convert syntax graphs to dataset batch
        #print(graphs_with_embeddings)
        pyg_data_batch = tg_data.Batch.from_data_list(graphs_with_embeddings)
        pyg_data_batch.to(torch.device('cpu'))
        #print(f"Graph Batch: {pyg_data_batch}")
        
        # Calculate syngnn output
        logits, attn = self.syngnn(torch.as_tensor(pyg_data_batch.x, dtype=torch.float), pyg_data_batch.edge_index, pyg_data_batch.edge_attr, pyg_data_batch.batch)

        # Calculate loss if true labels given
        if label_ids is not None:
            # Trim Bert label attention mask to graph token labels length
            token_mask_labels = []
            for sentence_idx, label_mask in enumerate(attention_mask_label):
                # Find SEP token id in sentence and get index
                sep_idx = label_ids.tolist()[sentence_idx].index(78)
                token_mask_labels_temp = label_mask.tolist()
                # Ignore SEP token
                token_mask_labels_temp[sep_idx] = 0
                token_mask_labels.extend(token_mask_labels_temp)
            token_mask_labels = torch.tensor(token_mask_labels)

            """
            for graph_idx, graph in enumerate(syntax_graphs):
                token_mask_labels_temp = []
                token_mask_labels_temp.append(torch.tensor([0]))
                token_mask_labels_temp.extend([attention_mask_label[graph_idx][1:len(graph.x)-1] for graph_idx, graph in enumerate(syntax_graphs)])
                padding_size = attention_mask_label.size()[0] - token_mask_labels_temp[0].size()[0]
                token_mask_labels_temp.extend(torch.zeros(1,padding_size))
                token_mask_labels.extend(token_mask_labels_temp)
            print(token_mask_labels)
            token_mask_labels = torch.cat(token_mask_labels, dim=0)
            """

            logits_view = logits.view(-1, self.num_labels)
            labels_view = label_ids.view(-1)

           

            # Loss function: do not count labels with index 0, that is tokens labelled with X (=ignore)
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Trim Bert labels to contain only graph token labels
            if attention_mask_label is not None:
                active_loss = token_mask_labels.view(-1) == 1
                #print(active_loss)
            #   active_logits = logits_view[active_loss]
                active_labels = labels_view[active_loss]
                #print(active_labels.size())
                #if (active_labels.size()[0] == 121):

                # print("Logits:")
                # print(logits.size())
                # print(logits_view.size())
                # print("Labels:")
                # print(label_ids.size())
                # print(active_labels.size())
                # print("Label att mask")
                # print(attention_mask_label)
                # print("Label token mask")
                # print(token_mask_labels)
                '''tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
                    print("Tokens sentence:")
                    for sentence in input_ids:
                        print(tokenizer.convert_ids_to_tokens(sentence))
                    #print(sentence)
                    print("Graph:")
                    for graph in graphs_with_embeddings:
                        print(graph)
                    print(f"Logits: {logits.size()}")'''
                try:
                    loss = loss_fct(logits_view, active_labels)
                    #print(loss)
                except:
                    print("Logits:")
                    print(logits.size())
                    print(logits_view.size())
                    print("Labels:")
                    print(label_ids.size())
                    print(active_labels.size())
                    print("Input Ids")
                    print(input_ids)
                    print(input_ids.size())
                    #print("Label att mask")
                    #print(attention_mask_label)
                    #print("Label token mask")
                    #print(token_mask_labels.size())
                    #print(active_loss)
                    #tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
                    #print("Tokens sentence:")
                    #for sentence in input_ids:
                    #    print(tokenizer.convert_ids_to_tokens(sentence))
                    #print(sentence)
                    #print("Graph:")
                    #for graph in graphs_with_embeddings:
                    #    print(graph)
            else:
                loss = loss_fct(logits_view, labels_view)
            return loss, logits
        else:
            return logits