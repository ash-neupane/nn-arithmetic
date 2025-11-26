import torch
import torch.nn.functional as F

from nn_arithmetic.model.language_model import GPTTransformer
from nn_arithmetic.tokenizer import ShakespeareTokenizer


def sample_from_model(
    model: GPTTransformer,
    tokenizer: ShakespeareTokenizer,
    prompt: str = "",
    max_length: int = 100,
    temperature: float = 1.0,
    top_k: int | None = None,
    device: str = "cpu",
) -> str:
    """
    Generate text from the language model.

    Args:
        model: Trained GPTTransformer model
        tokenizer: Tokenizer used during training
        prompt: Starting text (empty string for unconditional generation)
        max_length: Maximum number of tokens to generate
        temperature: Sampling temperature (higher = more random)
        top_k: If set, only sample from top k most likely tokens
        device: Device to run inference on

    Returns:
        Generated text as a string
    """
    model.eval()
    model = model.to(device)

    if prompt:
        tokens = tokenizer.encode(prompt).ids
    else:
        tokens = [0]

    tokens = torch.tensor(tokens, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        for _ in range(max_length):
            if tokens.size(1) >= model.max_seq_length:
                context = tokens[:, -model.max_seq_length :]
            else:
                context = tokens

            logits = model(context)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat([tokens, next_token], dim=1)

    generated_ids = tokens.squeeze(0).tolist()
    generated_text = tokenizer.decode(generated_ids)

    return generated_text


if __name__ == "__main__":
    print("To use this module:")
    print("  from nn_arithmetic.engine.generate import sample_from_model")
    print("  from nn_arithmetic.tokenizer import ShakespeareTokenizer")
    print("  tokenizer = ShakespeareTokenizer()")
    print("  text = sample_from_model(model, tokenizer, prompt='To be or not to be')")
