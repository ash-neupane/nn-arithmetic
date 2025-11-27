from dotenv import load_dotenv

from nn_arithmetic.engine import (
    train_modulo_adder,
    train_language_model,
    train_primality_classifier,
)

if __name__ == "__main__":
    load_dotenv()

    # train_modulo_adder(
    #     modulo_factor=89,
    #     epochs=1000,
    # )

    # train_language_model(epochs=5)

    train_primality_classifier(epochs=100)
