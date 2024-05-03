
from dvclive import Live

def save_model_dvc():
  with Live() as live:
    print("\033[35m", "Tracking model with DVC...", "\033[0m")

    live.log_artifact(
        str("model.keras"),
        type="model",
        name="mnist_model",
        desc="This is an example model trained on the MNIST dataset.",
        labels=["cv", "mnist", "model", "dvc", "keras"]
    )