import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import CustomEncoderLayer


class HybridTransformer(nn.Module):
    """Notebook type='time_region_transformer' path only."""

    def __init__(self,
                 n_rois=118,
                 n_timesteps=142,
                 d_model=128,
                 n_heads=4,
                 n_layers=4,           # must be even for time_region mode
                 n_classes=2,
                 dropout=0.3,
                 type="time_region_transformer"):
        super().__init__()

        if type != "time_region_transformer":
            raise ValueError(f"HybridTransformer requires type='time_region_transformer', got {type!r}")

        self.type = type
        self.n_layers = n_layers
        self.n_rois = n_rois
        self.n_timesteps = n_timesteps
        self.d_model = d_model

        # -------- Stage 1: time transformer (sequence = time) --------
        self.time_input_projection = nn.Linear(n_rois, d_model)
        self.time_pos_encoding = nn.Parameter(torch.randn(n_timesteps, d_model) * 0.1)
        self.time_to_region = nn.Linear(d_model, n_rois)

        # -------- Stage 2: region transformer (sequence = ROI) --------
        self.region_input_projection = nn.Linear(n_timesteps, d_model)
        self.region_pos_encoding = nn.Parameter(torch.randn(n_rois, d_model) * 0.1)

        assert n_layers % 2 == 0, "n_layers must be even for time_region_transformer"
        self.n_time_layers = n_layers // 2
        self.n_region_layers = n_layers - self.n_time_layers

        self.time_layers = nn.ModuleList([
            CustomEncoderLayer(d_model, n_heads, d_model * 2, dropout)
            for _ in range(self.n_time_layers)
        ])
        self.region_layers = nn.ModuleList([
            CustomEncoderLayer(d_model, n_heads, d_model * 2, dropout)
            for _ in range(self.n_region_layers)
        ])

        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        # applies Xavier on a fresh run. Calling apply here would Xavier-init twice.
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

        # ---- Stage 1: Time transformer on (B, T, R) ----
        x_time = self.time_input_projection(x)         # (B, T, d_model)
        x_time = x_time + self.time_pos_encoding[:T].unsqueeze(0)
        x_time = F.dropout(x_time, p=0.1, training=self.training)

        for layer in self.time_layers:
            if return_attention:
                x_time, attn = layer(
                    x_time, return_attention=True,
                    average_attn_weights=average_attn_weights
                )
                all_attention_weights.append(attn)
            else:
                x_time = layer(x_time, return_attention=False)

        # ---- Linear adapter: time -> region ----
        # x_time: (B, T, d_model) -> (B, T, R)
        x_tr = self.time_to_region(x_time)             # (B, T, R)
        x_region_in = x_tr.transpose(1, 2)             # (B, R, T)

        # ---- Stage 2: Region transformer ----
        x_region = self.region_input_projection(x_region_in)  # (B, R, d_model)
        x_region = x_region + self.region_pos_encoding[:R].unsqueeze(0)
        x_region = F.dropout(x_region, p=0.1, training=self.training)

        for layer in self.region_layers:
            if return_attention:
                x_region, attn = layer(
                    x_region, return_attention=True,
                    average_attn_weights=average_attn_weights
                )
                all_attention_weights.append(attn)
            else:
                x_region = layer(x_region, return_attention=False)

        x = x_region

        pooled = x.mean(dim=1)          # (B, d_model)
        logits = self.classifier(pooled)

        if return_attention:
            return logits, all_attention_weights
        else:
            return logits
