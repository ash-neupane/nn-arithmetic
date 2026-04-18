import torch
import torch.nn as nn


class PrimalityTransformer(nn.Module):
    """Transformer-based primality classifier for binary-encoded integers."""

    def __init__(
        self,
        num_bits: int = 64,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_bits = num_bits
        self.d_model = d_model

        # Project binary input to d_model dimensions
        self.input_projection = nn.Linear(1, d_model)

        # Positional encoding (which bit position)
        self.pos_encoding = nn.Embedding(num_bits, d_model)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2),  # Binary: prime or composite
        )

    def forward(self, x):
        """
        Args:
            x: Binary tensor of shape (batch_size, num_bits) with values in {0, 1}

        Returns:
            Logits of shape (batch_size, 2) for [composite, prime]
        """
        B, T = x.shape
        assert T == self.num_bits, f"Expected {self.num_bits} bits, got {T}"

        # Add feature dimension: (B, T) -> (B, T, 1)
        x = x.unsqueeze(-1)

        # Project to d_model: (B, T, 1) -> (B, T, d_model)
        x = self.input_projection(x)

        # Add positional encoding
        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_encoding(positions)

        # Transformer encoding
        x = self.transformer(x)

        # Pool across sequence (mean pooling)
        x = x.mean(dim=1)

        # Classification
        logits = self.classifier(x)

        return logits


class PrimalityMLP(nn.Module):
    """Simple MLP baseline for primality classification."""

    def __init__(
        self,
        num_bits: int = 64,
        hidden_dim: int = 512,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_bits = num_bits

        layers = []
        input_dim = num_bits

        for _ in range(num_layers - 1):
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 2))

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: Binary tensor of shape (batch_size, num_bits)

        Returns:
            Logits of shape (batch_size, 2)
        """
        return self.network(x)
