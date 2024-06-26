import dagshub.auth
import tensorflow as tf
import dagshub
import mlflow
import mlflow.tensorflow
import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import json
from train_parallel import main as train_parallel


def main():
  print(
    "\033[35m",
    r"""

    /\ "-./  \   /\ \       /\  __ \   /\  == \ /\  ___\
    \ \ \-./\ \  \ \ \____  \ \ \/\ \  \ \  _-/ \ \___  \
    \ \_\ \ \_\  \ \_____\  \ \_____\  \ \_\    \/\_____\
      \/_/  \/_/   \/_____/   \/_____/   \/_/     \/_____/
    ________________________________________________________
    --------------------------------------------------------
    """,
    "\033[0m")

  print("\033[35m", "Starting training...", "\033[0m")

  dagshub.auth.add_app_token(os.environ["MLFLOW_TRACKING_PASSWORD"])
  dagshub.init("mlops-presentation", "NoahLeu", mlflow=True)

  mlflow.start_run()

  # Activate autologging with MLflow
  # mlflow.tensorflow.autolog()
  # we can't use autologging because we need to train in parallel

  print("\033[35m", "Loading data...", "\033[0m")

  # Load MNIST data
  mnist = tf.keras.datasets.mnist
  (x_train, y_train), (x_test, y_test) = mnist.load_data()

  # Split data in half for faster training
  x_train, _, y_train, _ = train_test_split(x_train, y_train, test_size=0.5, random_state=42)

  # Split the training data for validation
  x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

  print("\033[35m", "Building model...", "\033[0m")

  # Define model (very simple minimal example)
  model1 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(64, activation='linear'),
    tf.keras.layers.Dense(10, activation='softmax')
  ])

  # Compile model
  model1.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
  model2.compile(optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])

  models = [model1, model2]

  metrics = train_parallel(models, x_train, y_train, x_val, y_val)

  val_acc1, val_loss1, val_acc2, val_loss2 = 0, 0, 0, 0
  for i, metric in enumerate(metrics):
    if i == 0:
      val_acc1, val_loss1 = metric['accuracy'], metric['loss']
    else:
      val_acc2, val_loss2 = metric['accuracy'], metric['loss']

  print("\033[35m", "Saving metrics...", "\033[0m")

  # Save metrics to a CSV file
  metrics = pd.DataFrame({
    'model': ['model1', 'model2'],
    'val_loss': [val_loss1, val_loss2],
    'val_acc': [val_acc1, val_acc2]
  })

  metrics.to_csv('metrics/metrics.csv', index=False)

  # Select the best model
  best_model = model1 if val_acc1 > val_acc2 else model2
  best_model_name = "model_1" if val_acc1 > val_acc2 else "model_2"

  print("\033[35m", "Best model: " + best_model_name, "\033[0m")

  print("\033[35m", "Saving trained model...", "\033[0m")

  # Saving
  best_model.save("model.h5")

  print("\033[35m", "Tracking model with DVC...", "\033[0m")

  print("\033[35m", "Logging metrics and evaluating model...", "\033[0m")

  # Log metrics
  pred_y = best_model.predict(x_test)
  pred_df = pd.DataFrame(pred_y, columns=[i for i in range(10)])
  pred_df['Pred'] = pred_df.idxmax(axis=1)
  pred_df['GT'] = y_test

  print("\033[35m", "Creating and saving confusion matrix...", "\033[0m")

  # Confusion matrix plot
  cm = confusion_matrix(pred_df['GT'], pred_df['Pred'])
  cm_map = ConfusionMatrixDisplay(confusion_matrix=cm)
  cm_map.plot()
  plt.savefig("metrics/confusion_matrix.png")

  # Log confusion matrix
  mlflow.log_artifact(
    "metrics/confusion_matrix.png",
    "confusion_matrix.png"
  )

  # Log metrics metrics.json
  metrics = {
    "accuracy": val_acc1,
    "loss": val_loss1
  }

  with open("metrics/metrics.json", "w") as f:
      json.dump(metrics, f)

  print("\033[35m", "Logging model to MLflow...", "\033[0m")
  # Save model in MLflow format
  mlflow.tensorflow.log_model(best_model, "model")

  mlflow.end_run()

  print("\033[32m", "Run finished successfully.", "\033[0m")


if __name__ == "__main__":
  main()