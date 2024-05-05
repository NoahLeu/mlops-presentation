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
import multiprocessing as mp
import json
import multiprocessing as mp

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
mlflow.tensorflow.autolog()

print("\033[35m", "Loading data...", "\033[0m")

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Split the training data for validation
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

print("\033[35m", "Building model...", "\033[0m")

# Define model (very simple minimal example)
model1 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='softmax'),
  tf.keras.layers.Dense(10, activation='relu')
])

model2 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(10, activation='softmax')
])


def train_and_log(model, x_train, y_train, x_val, y_val, experiment_name):
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # Train the model here and log metrics with MLflow
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        history = model.fit(x_train, y_train, epochs=5, validation_data=(x_val, y_val))

        # Log metrics with mlflow
        for epoch in range(len(history.history['accuracy'])):
            mlflow.log_metric('accuracy', history.history['accuracy'][epoch], step=epoch)
            mlflow.log_metric('val_accuracy', history.history['val_accuracy'][epoch], step=epoch)
            mlflow.log_metric('loss', history.history['loss'][epoch], step=epoch)
            mlflow.log_metric('val_loss', history.history['val_loss'][epoch], step=epoch)

        # save evaluation metrics
        val_loss, val_acc = model.evaluate(x_val, y_val)
        if experiment_name == "model_1":
            out_val_loss1.value = val_loss
            out_val_acc1.value = val_acc
        else:
            out_val_loss2.value = val_loss
            out_val_acc2.value = val_acc

out_val_loss1 = mp.Value('d', 0.0)
out_val_acc1 = mp.Value('d', 0.0)
out_val_loss2 = mp.Value('d', 0.0)
out_val_acc2 = mp.Value('d', 0.0)

p1 = mp.Process(target=train_and_log, args=(model1, x_train, y_train, x_val, y_val, "model_1"))
p2 = mp.Process(target=train_and_log, args=(model2, x_train, y_train, x_val, y_val, "model_2"))

p1.start()
p2.start()

p1.join()
p2.join()

val_loss1 = out_val_loss1.value
val_acc1 = out_val_acc1.value
val_loss2 = out_val_loss2.value
val_acc2 = out_val_acc2.value

# Save metrics to a CSV file
metrics = pd.DataFrame({
    'model': ['model1', 'model2'],
    'val_loss': [val_loss1, val_loss2],
    'val_acc': [val_acc1, val_acc2]
})

metrics.to_csv('metrics/metrics.csv', index=False)

# Select the best model
best_model = model1 if val_acc1 > val_acc2 else model2
best_model_name = "model1" if val_acc1 > val_acc2 else "model2"

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
