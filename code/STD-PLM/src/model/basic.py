import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
import math

class MLP(nn.Module):
    """Multi-Layer Perceptron.

    Reference:
        Attention Is All You Need.
        https://arxiv.org/pdf/1706.03762.pdf.

    """

    def __init__(
        self,
        in_features: int,
        hidden_dim: int,
        out_features: int,
        act_fn: Optional[nn.Module] = nn.GELU(),
        dropout : float = 0,
    ) -> None:
        super().__init__()

        self.fc1 = nn.Linear(in_features,hidden_dim)
        self.act = act_fn
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, out_features)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.dropout(self.act(hidden_states))
        hidden_states = self.fc2(hidden_states)

        return hidden_states
    
class spatialGCN(nn.Module):
    def __init__(self, sym_norm_Adj_matrix, in_channels, out_channels):
        super(spatialGCN, self).__init__()
        self.sym_norm_Adj_matrix = sym_norm_Adj_matrix  # (N, N)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.Theta = nn.Linear(in_channels, out_channels, bias=False)

    def forward(self, x):
        '''
        spatial graph convolution operation
        :param x: (batch_size, N, T, F_in)
        :return: (batch_size, N, T, F_out)
        '''
        batch_size, num_of_vertices, num_of_timesteps, in_channels = x.shape

        x = x.permute(0, 2, 1, 3).reshape((-1, num_of_vertices, in_channels))  # (b*t,n,f_in)

        return F.relu(self.Theta(torch.matmul(self.sym_norm_Adj_matrix, x)).reshape((batch_size, num_of_timesteps, num_of_vertices, self.out_channels)).transpose(1, 2))

class Prompt_pool(nn.Module):

    def __init__(
        self,
        emb_dim,
        pool_size=30,
        top_k=3,
        pp_dim = None,
        dropout=0
    ) -> None:
        super().__init__()

        if pp_dim is None:
            pp_dim = emb_dim
        v_dim = pp_dim

        self.key_pool = nn.Parameter(torch.randn(pool_size,emb_dim))
        self.value_pool = nn.Parameter(torch.randn(pool_size,v_dim))

        self.fc = nn.Linear(in_features=top_k*v_dim + emb_dim,out_features=emb_dim)

        self.top_k = top_k

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:

        '''
            hidden_states: [Batch,T,emb_dim]
            return: [Batch,T,emb_dim]
        '''
        B,T,D = hidden_states.shape

        hidden_states_norm = torch.nn.functional.normalize(hidden_states, p=2, dim=-1)
        key_pool_norm = torch.nn.functional.normalize(self.key_pool, p=2, dim=-1)

        weights = torch.matmul(hidden_states_norm,key_pool_norm.transpose(0,1)) #[Batch,T,pool_size]

        _,indices = torch.topk(weights,self.top_k) #[Batch,T,top_k]

        pool_states = self.value_pool[indices.view(-1),:].view(B,T,-1) #[Batch,T,top_k*emb_dim]

        hidden_states = torch.concat((hidden_states,pool_states),dim=-1)

        out = self.fc(hidden_states)

        key_pool_norm_topk = key_pool_norm[indices.view(-1),:].view(B,T,self.top_k,-1) #[Batch,T,top_k*emb_dim]
        sim_loss = (hidden_states_norm.unsqueeze(2)*key_pool_norm_topk).sum()/B

        return out,sim_loss
    

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor