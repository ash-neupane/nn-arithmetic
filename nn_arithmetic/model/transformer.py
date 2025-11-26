import torch
import torch.nn as nn


class ModuloTransformer(nn.Module):
    def __init__(
        self, modulo_factor: int, d_model: int = 64, nhead: int = 4, num_layers: int = 2, dim_feedforward: int = 128
    ):
        super().__init__()
        self.modulo_factor = modulo_factor
        self.d_model = d_model

        self.embedding = nn.Embedding(modulo_factor, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(2, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, modulo_factor)

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding.unsqueeze(0)

        x = self.transformer(x)
        x = x.mean(dim=1)

        return self.output_projection(x)
