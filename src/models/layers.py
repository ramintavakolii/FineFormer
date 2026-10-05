import torch.nn as nn


class CustomEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, dim_feedforward, dropout):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Feed forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, x, return_attention=False, average_attn_weights=False):
        # Self-attention
        if return_attention:
            attn_output, attn_weights = self.self_attn(
                x, x, x, average_attn_weights=average_attn_weights
            )
        else:
            attn_output, _ = self.self_attn(x, x, x)
            attn_weights = None

        # Add & norm
        x = self.norm1(x + self.dropout1(attn_output))

        # Feed forward
        ff_output = self.linear2(self.dropout(self.activation(self.linear1(x))))

        # Add & norm
        x = self.norm2(x + self.dropout2(ff_output))

        if return_attention:
            return x, attn_weights
        else:
            return x
