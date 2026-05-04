import joblib
import matplotlib.pyplot as plt
import os

# CONFIG (ensure this matches your training script)
MODEL_DIR = "models"
HISTORY_FILE = os.path.join(MODEL_DIR, "dnn_history.joblib")

# 1. Load the training history
try:
    history = joblib.load(HISTORY_FILE)
except FileNotFoundError:
    print(f"Error: History file not found at {HISTORY_FILE}. Please run the training script first.")
    exit()

# 2. Extract metrics
# The keys in the history dictionary are typically 'loss', 'accuracy', 'val_loss', 'val_accuracy'
train_loss = history['loss']
val_loss = history['val_loss']
train_accuracy = history['accuracy']
val_accuracy = history['val_accuracy']
epochs = range(1, len(train_loss) + 1) # Epochs start from 1

## 3. Plot Loss vs. Epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'r-', label='Training Loss')
plt.plot(epochs, val_loss, 'b--', label='Validation Loss')
plt.title('Loss Curve: Training vs. Validation')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

## 4. Plot Accuracy vs. Epochs
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'r-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'b--', label='Validation Accuracy')
plt.title('Accuracy Curve: Training vs. Validation')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Optional: Print final performance metrics
print("-" * 30)
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {val_loss[-1]:.4f}")
print(f"Final Training Accuracy: {train_accuracy[-1]:.4f}")
print(f"Final Validation Accuracy: {val_accuracy[-1]:.4f}")
print("-" * 30)