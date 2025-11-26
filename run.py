from nn_arithmetic.trainer.train import train_modulo_adder
from dotenv import load_dotenv


if __name__=="__main__":
    load_dotenv()
    train_modulo_adder(
        modulo_factor=89,
        epochs=1000,
    )
