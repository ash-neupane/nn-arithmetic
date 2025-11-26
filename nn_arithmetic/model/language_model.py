import torch
import torch.nn as nn


class GPTTransformer(nn.Module):
    def __init__(
        self, vocab_size: int, d_model: int = 128, nhead: int = 4, num_layers: int = 4, dim_feedforward: int = 512, max_seq_length: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Embedding(max_seq_length, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        B, T = x.shape

        positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.embedding(x) + self.pos_encoding(positions)

        if mask is None:
            mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        x = self.transformer(x, x, tgt_mask=mask, memory_mask=mask)
        logits = self.output_projection(x)

        return logits
