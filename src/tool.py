from __future__ import annotations
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
FIG_DIR = Path("/content/drive/MyDrive/California-housing/figures")

def get_metrics(y, y_pred, model_name, data):
  return {
      "model": model_name,
      "data": data,
      "r^2": r2_score(y, y_pred),
      "mae": mean_absolute_error(y, y_pred),
      "mse": mean_squared_error(y, y_pred),
      "rmse": mean_squared_error(y, y_pred) ** 0.5
  }

def plot_true_vs_pred(test, pred, name):
  plt.figure(figsize=(6,6))
  plt.scatter(test, pred, alpha=0.3)
  plt.plot([min(test), max(test)], [min(test), max(test)], 'r--')  # y=x line
  plt.xlabel('True')
  plt.ylabel('Predicted')
  plt.title(f"True vs. Predicted {name}")
  plt.savefig(f"{FIG_DIR}/True vs. Predicted {name}.png", dpi=300, bbox_inches="tight")
  plt.show()
