import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

log_path = 'runs/detect/train14/results.csv'

df = pd.read_csv(log_path)

df.columns = df.columns.str.strip()

print(df.columns.tolist())

# Create subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# === Subplot 1: Loss ===
axes[0].plot(df['epoch'], df['train/box_loss'], label='Train Box Loss')
axes[0].plot(df['epoch'], df['val/box_loss'], label='Val Box Loss')
axes[0].set_title("Training vs Validation Loss")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(True)

# === Subplot 2: Accuracy ===
axes[1].plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5')
axes[1].plot(df['epoch'], df['metrics/precision(B)'], label='Precision')
axes[1].plot(df['epoch'], df['metrics/recall(B)'], label='Recall')
axes[1].set_title("Validation Metrics")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Score")
axes[1].legend()
axes[1].grid(True)

# Layout and show
plt.tight_layout()
plt.show()


# Get the latest training folder
latest_run = sorted(glob('runs/detect/train*'))[-1]
df = pd.read_csv(os.path.join(latest_run, 'results.csv'))
