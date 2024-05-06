import mlflow
import mlflow.tensorflow
from return_thread import ReturnValueThread

def train_and_log(model, run_name, x_train, y_train, x_val, y_val):
  with mlflow.start_run(run_name=run_name, nested=True) as run:
    # Train and evaluate the model
    print ("\033[35m", f"Training {run_name}...", "\033[0m")
    history = model.fit(x_train, y_train, epochs=3, validation_data=(x_val, y_val))
    print ("\033[35m", f"Evaluating {run_name}...", "\033[0m")
    val_loss, val_acc = model.evaluate(x_val, y_val)
    # Log metrics
    for epoch in range(len(history.history['loss'])):
      epoch_metrics = {
        "train_loss": history.history['loss'][epoch],
        "train_acc": history.history['accuracy'][epoch],
        "val_loss": history.history['val_loss'][epoch],
        "val_acc": history.history['val_accuracy'][epoch]
      }

      mlflow.log_metrics(epoch_metrics, step=epoch, run_id=run.info.run_id)

    final_metrics = {
      "accuracy": val_acc,
      "loss": val_loss
    }

    mlflow.log_metrics(final_metrics, run_id=run.info.run_id)

    # Save model in MLflow format
    # mlflow.tensorflow.log_model(model, "model")

    return final_metrics

def main(models, x_train, y_train, x_val, y_val):
  threads = []
  for i, model in enumerate(models):
    t = ReturnValueThread(target=train_and_log, args=(model, f"model_{i+1}", x_train, y_train, x_val, y_val))
    threads.append(t)

  for t in threads:
    t.start()

  metrics = []
  for t in threads:
    metric_pair = t.join()
    metrics.append(metric_pair)

  # form: [{'accuracy': 0.5, 'loss': 0.5}, {'accuracy': 0.5, 'loss': 0.5}]
  return metrics