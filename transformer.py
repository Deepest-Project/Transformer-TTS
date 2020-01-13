import torch
import torch.nn as nn
import torch.nn.functional as F
from init_layer import *


class MultiheadAttentionCat(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super(MultiheadAttentionCat, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear = nn.Linear(2 * d_model, d_model)

    def forward(self,
                query,
                key,
                value,
                key_padding_mask=None,
                need_weights=True,
                attn_mask=None):
        output, weights = self.self_attn(query,
                                         key,
                                         value,
                                         key_padding_mask,
                                         need_weights,
                                         attn_mask)

        output = self.linear(torch.cat([output, query], dim=-1))

        return output, weights


class TransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttentionCat(d_model, nhead, dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward, w_init_gain=activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src,
                                 src,
                                 src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(TransformerDecoderLayer, self).__init__()
        self.self_attn = MultiheadAttentionCat(d_model, nhead, dropout=dropout)
        self.multihead_attn = MultiheadAttentionCat(d_model,
                                                    nhead,
                                                    dropout=dropout)

        self.linear1 = Linear(d_model, dim_feedforward, w_init_gain=activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self,
                tgt,
                memory,
                tgt_mask=None,
                memory_mask=None,
                tgt_key_padding_mask=None,
                memory_key_padding_mask=None):
        tgt2, _ = self.self_attn(tgt,
                                 tgt,
                                 tgt,
                                 attn_mask=tgt_mask,
                                 key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2, alignments = self.multihead_attn(tgt,
                                               memory,
                                               memory,
                                               attn_mask=memory_mask,
                                               key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(F.relu(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt, alignments


class FFTblock(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 dim_feedforward=2048,
                 dropout=0.1,
                 activation="relu"):
        super(FFTblock, self).__init__()
        self.self_attn = MultiheadAttentionCat(d_model, nhead, dropout=dropout)

        self.conv1 = Conv1d(d_model,
                            dim_feedforward,
                            kernel_size=3,
                            padding=1,
                            w_init_gain=activation)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = Conv1d(dim_feedforward, d_model, kernel_size=3, padding=1)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, _ = self.self_attn(src,
                                 src,
                                 src,
                                 attn_mask=src_mask,
                                 key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.dropout(F.relu(self.conv1(src.permute(1, 2, 0).contiguous())))
        src2 = self.conv2(src2).permute(2, 0, 1).contiguous()
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.pow(10000, torch.arange(0, d_model, 2).float() / d_model)
        
        self.pe[:, 0::2] = torch.sin(position / div_term)
        self.pe[:, 1::2] = torch.cos(position / div_term)

    def forward(self, x):
        return x + x.new_tensor(self.pe[:x.size(0)].unsqueeze(1),
                                dtype=torch.float)