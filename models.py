# %%
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

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        
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

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask_label is not None:
                active_loss = attention_mask_label.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss, logits
        else:
            return logits
# %%
class SynGNNLayer(nn.Module):
    """
    SynGNN Pytorch module
    based on Pytorch TransformerEncoderLayer implementing the architecture in paper “Attention Is All You Need”. 
    
    """
    def __init__(self, dim_in, dim_hdn, dim_out, num_heads, dim_feedforward=2048, dropout=0.1, activation="relu"):
        """
        :param dim_in: input dimension
        :param dim_hdn: hidden nodes dimension
        :param dim_out: output dimension
        """
        super(SynGNNLayer, self).__init__()
        # Graph attention sublayer
        self.graph_attn = tg_nn.GATv2Conv(in_channels=dim_in, out_channels=dim_hdn , heads=num_heads)
        self.linear1 = tg_nn.Linear(dim_hdn*num_heads, dim_hdn*num_heads)
        self.linear2 = tg_nn.Linear(dim_hdn*num_heads, dim_hdn*num_heads)
        self.linear_classifier = tg_nn.Linear(dim_hdn*num_heads, dim_out)

        self.norm0 = tg_nn.LayerNorm(dim_in)
        self.norm1 = tg_nn.LayerNorm(dim_hdn*num_heads)
        self.norm2 = tg_nn.LayerNorm(dim_hdn*num_heads)
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

    def forward(self, x, edge_index, batch):
        r"""Pass the input through the encoder layer.
        Args:
            x: node features
            edge_index: graph edges
            batch: current batch
        """
        # Graph attention sublayer
        src = self.norm0(x)
        print(f"Input: {src.size()}")
        src2, att = self.graph_attn(src, edge_index, return_attention_weights = True)

        print(f"Graph Att Output src2: {src2.size()}")
        src = src + self.dropout1(src2)
        #print(f" After adding layers: {src.size()}")
        src = self.norm1(src)

        # Feed-Forward-Network sublayer
        src2 = self.linear2(self.dropout0(self.activation(self.linear1(src))))

        print(f" After linear layer 2: {src2.size()}")
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        src = self.linear_classifier(src)
        src = self.norm3(src)
        print(f" After linear output layer: {src.size()}")
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

    def forward(self, x, edge_index, batch):
        r"""Pass the input through the encoder layers in turn.
        Args:
            x: node features
            edge_index: graph edges
            batch: current batch
        """
        output = x
        attns = []

        for layer in self.layers:
            output, attn = layer(x, edge_index, batch)
            attns.append(attn)
        #attns = torch.stack(attns)

        if self.norm is not None:
            output = self.norm(output)

        return output, attns
    
    def add_bert_embeddings_to_graph(self, ptg_graph, pt_embeddings, sentence_graph_idx_map):

        for sent_idx, pt_embedding in enumerate(pt_embeddings):
            embedding = pt_embedding.detach().clone()
            if sent_idx in sentence_graph_idx_map:
                graph_idx = sentence_graph_idx_map[sent_idx]
                #print(ptg_graph.x[graph_idx])
                #ptg_graph.x.resize(ptg_graph.num_nodes, embedding.size())
                ptg_graph.x[graph_idx] = torch.tensor(embedding,dtype=torch.float32)


        return ptg_graph


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])        
# %%
class SynBertForNer(nn.Module):
    def __init__(self, bert_config, num_node_features, num_labels, num_att_heads, num_layers, device_str=None):

        super(SynBertForNer, self).__init__()
        self.num_labels = num_labels

        self.bert = BertModel(bert_config)
        self.gnn_layer = SynGNNLayer(dim_in=num_node_features, dim_hdn = num_node_features, dim_out = num_labels, num_heads = num_att_heads)
        self.syngnn = SynGNN(self.gnn_layer, num_layers = num_layers)

    def forward(self, input_ids, syntax_graphs, sentence_graph_idx_maps, token_type_ids=None, attention_mask=None, labels=None,valid_ids=None,attention_mask_label=None):
        
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
            graphs_with_embeddings.append(self.syngnn.add_bert_embeddings_to_graph(syntax_graphs[batch_idx], valid_output[batch_idx],sentence_graph_idx_maps[batch_idx]))

        # Convert syntax graphs to dataset batch
        pyg_data_batch = tg_data.Batch.from_data_list(graphs_with_embeddings)
        pyg_data_batch.to(torch.device('cpu'))
        # Calculate syngnn output
        logits, attn = self.syngnn(torch.tensor(pyg_data_batch.x,dtype=torch.float), pyg_data_batch.edge_index, pyg_data_batch.batch)

        # Calculate loss if true labels given
        if labels is not None:
            logits_view = logits.view(-1, self.num_labels)
            labels_view = labels.view(-1)
            print("Logits:")
            print(logits.size())
            print(logits_view.size())
            print("Labels:")
            print(labels.size())
            print(labels_view.size())
            # Loss function: do not count labels with index 0, that is tokens labelled with X (=ignore)
            loss_fct = nn.CrossEntropyLoss(ignore_index=0)
            # Only keep active parts of the loss
            if attention_mask_label is not None:
                   active_loss = attention_mask_label.view(-1) == 1
                   active_logits = logits.view(-1, self.num_labels)[active_loss]
                   active_labels = labels.view(-1)[active_loss]
                   loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits_view, labels_view)
            return loss, logits
        else:
            return logits