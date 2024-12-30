import torch
import torch.nn as nn
import numpy as np

class ScaledDotProductAttention(nn.Module):
    def __init__(self, head: int, d_model: int):
        super(ScaledDotProductAttention, self).__init__()

        self.d_model = d_model
        self.d_q = d_model // head
        self.d_kv = d_model // head
        self.head = head

        self.fc_q = nn.Linear(d_model, head * self.d_q)
        self.fc_k = nn.Linear(d_model, head * self.d_kv)
        self.fc_v = nn.Linear(d_model, head * self.d_kv)

    def forward(self, queries, keys, values, attention_mask=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        # Đảm bảo các tensor có dtype là float
        queries = queries.float()
        keys = keys.float()
        values = values.float()

        # Tính toán Q, K, V
        q = self.fc_q(queries).view(b_s, nq, self.head, self.d_q).permute(0, 2, 1, 3)  # (b_s, h, nq, d_q)
        k = self.fc_k(keys).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 3, 1)    # (b_s, h, d_kv, nk)
        v = self.fc_v(values).view(b_s, nk, self.head, self.d_kv).permute(0, 2, 1, 3)  # (b_s, h, nk, d_kv)

        # Tính attention scores
        att = torch.matmul(q, k) / np.sqrt(self.d_kv)  # (b_s, h, nq, nk)
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(1).to(att.dtype)  # Đảm bảo cùng dtype với att
            att = att.masked_fill(attention_mask == 0, -1e4)  # Đặt giá trị -1e4 cho các vị trí bị che

        # Softmax và weighted sum
        att = torch.softmax(att, dim=-1)  # (b_s, h, nq, nk)
        output = torch.matmul(att, v).permute(0, 2, 1, 3).reshape(b_s, -1, self.d_model)  # (b_s, nq, d_model)

        return output
