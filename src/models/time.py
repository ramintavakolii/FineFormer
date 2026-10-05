import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import CustomEncoderLayer


class TimeTransformer(nn.Module):
    def __init__(self,
                 n_rois=118,
                 n_timesteps=142,
                 d_model=128,
                 n_heads=4,
                 n_layers=4,
                 n_classes=2,
                 dropout=0.3,
                 type="time_transformer"):
        super().__init__()

        if type != "time_transformer":
            raise ValueError(f"TimeTransformer requires type='time_transformer', got {type!r}")

        self.type = type
        self.n_layers = n_layers
        self.n_rois = n_rois
        self.n_timesteps = n_timesteps
        self.d_model = d_model

        # sequence = time, features = ROIs
        self.input_projection = nn.Linear(n_rois, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(n_timesteps, d_model) * 0.1)

        self.transformer_layers = nn.ModuleList([
            CustomEncoderLayer(d_model, n_heads, d_model * 2, dropout)
            for _ in range(n_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        # self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x, return_attention=False, average_attn_weights=True):
        """
        x: (batch_size, n_timesteps, n_rois) == (B, T, R)
        """
        B, T, R = x.shape
        all_attention_weights = []

        # (B, T, R) -> (B, T, d_model)
        x = self.input_projection(x)
        x = x + self.pos_encoding[:T].unsqueeze(0)
        x = F.dropout(x, p=0.1, training=self.training)

        for layer in self.transformer_layers:
            if return_attention:
                x, attn = layer(
                    x, return_attention=True,
                    average_attn_weights=average_attn_weights
                )
                all_attention_weights.append(attn)
            else:
                x = layer(x, return_attention=False)

        pooled = x.mean(dim=1)          # (B, d_model)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, all_attention_weights
        else:
            return logits
