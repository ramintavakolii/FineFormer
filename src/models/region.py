import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import CustomEncoderLayer


class CompactfMRITransformer(nn.Module):
    """Region-only CompactfMRITransformer (notebook type='region_transformer')."""

    def __init__(self,
                 n_rois=118,
                 n_timesteps=142,
                 d_model=128,
                 n_heads=4,
                 n_layers=4,
                 n_classes=2,
                 dropout=0.3,
                 type="region_transformer"):
        super().__init__()

        self.type = type
        if self.type != "region_transformer":
            raise ValueError(
                "Region module implements only region_transformer. "
                f"Got type={self.type!r}."
            )

        # Input projection and Learnable positional encoding
        # sequence = ROIs, features = time  (notebook region path)
        self.input_projection = nn.Linear(n_timesteps, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(n_rois, d_model) * 0.1)

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

        # x shape: (batch_size, 142, 118)
        batch_size, time_point, region_point = x.shape

        # Region path (notebook):
        # (B, T, R) -> (B, R, T) -> project -> add pos -> dropout 0.1
        x = torch.transpose(x, 1, 2)
        x = self.input_projection(x)  # (batch, 118, d_model)
        x = x + self.pos_encoding[:region_point].unsqueeze(0)

        # Apply dropout to input (hardcoded p=0.1 in notebooks)
        x = F.dropout(x, p=0.1, training=self.training)

        # Store attention weights from all layers
        all_attention_weights = []

        # Pass through transformer layers
        for layer in self.transformer_layers:
            if return_attention:
                x, attn_weights = layer(
                    x, return_attention=True, average_attn_weights=average_attn_weights
                )
                all_attention_weights.append(attn_weights)
            else:
                x = layer(x, return_attention=False)

        # Global average pooling over sequence dimension
        pooled = x.mean(dim=1)  # (batch, d_model)

        # Classification
        logits = self.classifier(pooled)  # (batch, n_classes)

        if return_attention:
            return logits, all_attention_weights
        else:
            return logits


# Alias used by factory / configs
RegionTransformer = CompactfMRITransformer
