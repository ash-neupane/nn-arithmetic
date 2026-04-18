# tiny_shakespeare

Karpathy's tiny-shakespeare character corpus. Consumed by
`nn_arithmetic.dataset.tiny_shakespeare` for language-model training.

## Expected files

- `input.txt` — raw corpus (~1.1 MB)
- `train.pt`, `val.pt` — tokenized splits produced locally

## Download

From the repo root:

```
bash scripts/download_tiny_shakespeare.sh
bash scripts/tokenize_shakespeare.sh
```

Source: https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
