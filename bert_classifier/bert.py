from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *
import pprint


class BertSelfAttention(nn.Module):
  def __init__(self, config):
    super().__init__()

    self.config = config
    
    self.lora = False

    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize the linear transformation layers for key, value, query
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)
    # this dropout is applied to normalized attention scores following the original implementation of transformer
    # although it is a bit unusual, we empirically observe that it yields better performance
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


  def lora_init(self, lora_config):
    self.lora = True
    self.lora_config = lora_config
    self.lora_scaling = self.config.hidden_size / self.lora_config.lora_rank
    
    self.query_lora = nn.Sequential(
      nn.Dropout(self.lora_config.lora_dropout),
      nn.Linear(self.config.hidden_size, self.lora_config.lora_rank, bias=False),
      nn.Linear(self.lora_config.lora_rank, self.all_head_size, bias=False)
    )

    self.key_lora = nn.Sequential(
      nn.Dropout(self.lora_config.lora_dropout),
      nn.Linear(self.config.hidden_size, self.lora_config.lora_rank, bias=False),
      nn.Linear(self.lora_config.lora_rank, self.all_head_size, bias=False)
    )

    self.value_lora = nn.Sequential(
      nn.Dropout(self.lora_config.lora_dropout),
      nn.Linear(self.config.hidden_size, self.lora_config.lora_rank, bias=False),
      nn.Linear(self.lora_config.lora_rank, self.all_head_size, bias=False)
    )


  def transform(self, x, linear_layer, lora_module=None):
    # the corresponding linear_layer of k, v, q are used to project the hidden_state (x)
    bs, seq_len = x.shape[:2]
    proj = linear_layer(x)
    if self.lora:
      lora_proj = lora_module(x)
      proj = proj + lora_proj * self.lora_scaling
    # next, we need to produce multiple heads for the proj 
    # this is done by spliting the hidden state to self.num_attention_heads, each of size self.attention_head_size
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    # by proper transpose, we have proj of [bs, num_attention_heads, seq_len, attention_head_size]
    proj = proj.transpose(1, 2)
    return proj

  def attention(self, key, query, value, attention_mask):
    # each attention is calculated following eq (1) of https://arxiv.org/pdf/1706.03762.pdf
    # attention scores are calculated by multiply query and key 
    # and get back a score matrix S of [bs, num_attention_heads, seq_len, seq_len]
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # Note again: in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number 

    # normalize the scores
    # multiply the attention scores to the value and get back V'
    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]

    ### TODO
    # raise NotImplementedError
    # something wrong with the following code
    x = torch.matmul(query, key.transpose(-1, -2))
    # print(f'Q@K.shape: {x.shape}')
    # print(f'V.shape: {value.shape}')
    x = x / math.sqrt(key.size(-1))
    x = x.masked_fill(attention_mask < -1, -1e9)
    # print(attention_mask)
    x = F.softmax(x, dim=-1)
    x = torch.matmul(x, value)
    # concat back
    x = x.transpose(1, 2)
    x = x.reshape(x.size(0), x.size(1), -1)
    return x



  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: [bs, seq_len, hidden_state]
    attention_mask: [bs, 1, 1, seq_len]
    output: [bs, seq_len, hidden_state]
    """
    # first, we have to generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    if self.lora:
      key_layer = self.transform(hidden_states, self.key, self.key_lora)
      value_layer = self.transform(hidden_states, self.value, self.value_lora)
      query_layer = self.transform(hidden_states, self.query, self.query_lora)
    else:
      key_layer = self.transform(hidden_states, self.key)
      value_layer = self.transform(hidden_states, self.value)
      query_layer = self.transform(hidden_states, self.query)
    # calculate the multi-head attention 
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value


class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # multi-head attention
    self.config = config
    self.lora = False
    self.prompt = False

    self.self_attention = BertSelfAttention(config)
    # add-norm
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # another add-norm
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)


  def lora_init(self, lora_config):
    self.lora = True
    self.lora_config = lora_config
    self.lora_scaling_hidden = self.config.hidden_size / self.lora_config.lora_rank
    self.lora_scaling_interm = self.config.intermediate_size / self.lora_config.lora_rank

    self.attention_dense_lora = nn.Sequential(
      nn.Dropout(lora_config.lora_dropout),
      nn.Linear(self.config.hidden_size, self.lora_config.lora_rank, bias=False),
      nn.Linear(self.lora_config.lora_rank, self.config.hidden_size, bias=False)
    )

    self.interm_dense_lora = nn.Sequential(
      nn.Dropout(lora_config.lora_dropout),
      nn.Linear(self.config.hidden_size, self.lora_config.lora_rank, bias=False),
      nn.Linear(self.lora_config.lora_rank, self.config.intermediate_size, bias=False)
    )

    self.out_dense_lora = nn.Sequential(
      nn.Dropout(lora_config.lora_dropout),
      nn.Linear(self.config.intermediate_size, self.lora_config.lora_rank, bias=False),
      nn.Linear(self.lora_config.lora_rank, self.config.hidden_size, bias=False)
    )

    self.self_attention.lora_init(lora_config)
    
    
  def prompt_init(self, prompt_config, acumulated_prompt_length=0):
    self.prompt = True
    self.prompt_config = prompt_config
    self.single_prompt_length = prompt_config.single_prompt_length
    self.acumulated_prompt_length = acumulated_prompt_length + self.single_prompt_length
    
    self.prompt_tensor = nn.Parameter(torch.randn(1, self.single_prompt_length, self.config.hidden_size))
    # prompt tuning
    self.prompt_tensor.requires_grad = True


  def add_norm(self, input, output, dense_layer, dropout, ln_layer, lora_module=None, lora_scaling=None):
    """
    this function is applied after the multi-head attention layer or the feed forward layer
    input: the input of the previous layer
    output: the output of the previous layer
    dense_layer: used to transform the output
    dropout: the dropout to be applied 
    ln_layer: the layer norm to be applied
    """
    # Hint: Remember that BERT applies to the output of each sub-layer, before it is added to the sub-layer input and normalized 
    ### TODO
    # raise NotImplementedError
    if self.lora:
      x = dense_layer(output) + lora_module(output) * lora_scaling
      x = input + dropout(x)
    else:
      x = input + dropout(dense_layer(output))
    x = ln_layer(x)
    return x


  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    as shown in the left of Figure 1 of https://arxiv.org/pdf/1706.03762.pdf 
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the input and output of the multi-head attention layer
    3. a feed forward layer
    4. a add-norm that takes the input and output of the feed forward layer
    """
    ### TODO
    # raise NotImplementedError
    if self.prompt:
      # print(f'hidden_states.shape: {hidden_states.shape}')
      # print(f'attention_mask.shape: {attention_mask.shape}')
      # print(f'prompt_tensor.shape: {self.prompt_tensor.shape}')
      # example of sizes:
      # hidden_states.shape: torch.Size([8, 41, 768]) [batch_size, seq_len, hidden_size]
      # attention_mask.shape: torch.Size([8, 1, 1, 41]) [batch_size, 1, 1, seq_len]
      # prompt_tensor.shape: torch.Size([1, 2, 768]) [1, single_prompt_length, hidden_size]
      hidden_states = torch.cat([self.prompt_tensor.repeat(hidden_states.size(0), 1, 1), hidden_states], dim=1)
      attention_mask = torch.cat([torch.ones(attention_mask.size(0), 1, 1, self.acumulated_prompt_length, device=attention_mask.device), attention_mask], dim=-1)
      # print(f'attention_mask.shape: {attention_mask.shape}')
    x = self.self_attention(hidden_states, attention_mask)
      
    # just merge them
    if self.lora:
      x = self.add_norm(hidden_states, x, self.attention_dense, self.attention_dropout, self.attention_layer_norm, self.attention_dense_lora, self.lora_scaling_hidden)
      x_interm = self.interm_dense(x) + self.interm_dense_lora(x) * self.lora_scaling_hidden
      x_interm = self.interm_af(x_interm)
      x = self.add_norm(x, x_interm, self.out_dense, self.out_dropout, self.out_layer_norm, self.out_dense_lora, self.lora_scaling_interm)
    else:
      x = self.add_norm(hidden_states, x, self.attention_dense, self.attention_dropout, self.attention_layer_norm)
      x_interm = self.interm_af(self.interm_dense(x))
      x = self.add_norm(x, x_interm, self.out_dense, self.out_dropout, self.out_layer_norm)
      
    return x


class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    self.lora = False
    self.prompt = False

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.pooler_af = nn.Tanh()

    self.init_weights()


  def lora_init(self, lora_config):
    self.lora = True
    self.lora_config = lora_config
    self.lora_scaling = self.config.hidden_size / self.lora_config.lora_rank

    self.pooler_dense_lora = nn.Sequential(
      nn.Dropout(lora_config.lora_dropout),
      nn.Linear(self.config.hidden_size, self.lora_config.lora_rank, bias=False),
      nn.Linear(self.lora_config.lora_rank, self.config.hidden_size, bias=False)
    )

    for i, layer_module in enumerate(self.bert_layers):
      layer_module.lora_init(lora_config)
      
      
  def prompt_init(self, prompt_config):
    self.prompt = True
    self.prompt_config = prompt_config
    self.single_prompt_length = prompt_config.single_prompt_length
    self.total_prompt_length = 0
    
    for i, layer_module in enumerate(self.bert_layers):
      layer_module.prompt_init(prompt_config, self.total_prompt_length)
      self.total_prompt_length += self.single_prompt_length


  def embed(self, input_ids):
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # Get word embedding from self.word_embedding into input_embeds.
    inputs_embeds = None
    ### TODO
    # raise NotImplementedError
    inputs_embeds = self.word_embedding(input_ids)
    


    # Get position index and position embedding from self.pos_embedding into pos_embeds.
    pos_ids = self.position_ids[:, :seq_length]

    pos_embeds = None
    ### TODO
    # raise NotImplementedError
    pos_embeds = self.pos_embedding(pos_ids)


    # Get token type ids, since we are not consider token type, just a placeholder.
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # Add three embeddings together; then apply embed_layer_norm and dropout and return.
    ### TODO
    # raise NotImplementedError
    x = inputs_embeds + pos_embeds + tk_type_embeds
    x = self.embed_layer_norm(x)
    x = self.embed_dropout(x)
    return x


  def encode(self, hidden_states, attention_mask):
    """
    hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
    attention_mask: [batch_size, seq_len]
    """
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module.forward(hidden_states, extended_attention_mask)

    return hidden_states

  def forward(self, input_ids, attention_mask):
    """
    input_ids: [batch_size, seq_len], seq_len is the max length of the batch
    attention_mask: same size as input_ids, 1 represents non-padding tokens, 0 represents padding tokens
    """
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    if self.prompt:
      first_tk = sequence_output[:, self.total_prompt_length]
    else:
      first_tk = sequence_output[:, 0]
      
    if self.lora:
      first_tk = self.pooler_dense(first_tk) + self.pooler_dense_lora(first_tk) * self.lora_scaling
    else:
      first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)

    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
