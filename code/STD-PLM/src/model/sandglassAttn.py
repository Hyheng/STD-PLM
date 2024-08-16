import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.position import PositionalEncoding

class SAG(nn.Module):
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()

        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim

        self.hyper_nodes = nn.Parameter(torch.randn(1,sag_tokens,sag_dim))
        #self.pe = nn.Identity()
        self.pe = PositionalEncoding(num_hiddens=sag_dim,dropout=dropout,max_len=1024)

        self.emc_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout)
        self.dec_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout,vdim=emb_dim)

        self.enc_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)
        self.dec_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)

        self.x_fc = nn.Linear(in_features=emb_dim,out_features=sag_dim)


        self.en_ln = nn.LayerNorm(emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)
        

    def encode(self,x):
        #x(B,N,D)
        B,N,H = x.shape

        kv = self.x_fc(x)

        q = self.pe(self.hyper_nodes)

        out,attn_weights = self.emc_mha(query=q.repeat(B,1,1),key=self.pe(kv),value=kv) #B,N',D

        out = self.enc_fc(out)

        out = self.en_ln(out)

        return out,attn_weights

    def decode(self,hidden_state,x):
        #hidden_state(B,N',D)
        B,_,_ = hidden_state.shape

        q = self.pe(self.x_fc(x))
        k = self.pe(self.hyper_nodes)
        v = hidden_state

        out,_ = self.dec_mha(query=q,key=k.repeat(B,1,1),value=v) #B,N,H

        out = self.dec_fc(out)

        out = self.de_ln(out)

        return out
