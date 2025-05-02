import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Ensure plots directory exists
os.makedirs('./plots', exist_ok=True)

# Data: Sample Accuracy and Inference Time (for CPU & TPU)
labels = ['Lying', 'Sitting', 'Standing', 'Walking']
cpu_accuracy = [82.11, 61.47, 58.65, 91.84]
tpu_accuracy = [78.95, 59.63, 49.04, 92.85]
cpu_time = [0.1543, 0.1543, 0.1543, 0.1543]  # Avg Inference Time for CPU (constant for simplicity)
tpu_time = [0.0024, 0.0024, 0.0024, 0.0024]  # Avg Inference Time for TPU (constant for simplicity)

# Data for Heatmap
accuracy_data = np.array([cpu_accuracy, tpu_accuracy])
time_data = np.array([cpu_time, tpu_time])

# Plot Accuracy Heatmap
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(accuracy_data, annot=True, cmap="Blues", fmt=".2f", xticklabels=["CPU", "TPU"], yticklabels=labels)
ax.set_title("Heatmap: Per-Class Accuracy Comparison (CPU vs TPU)")
plt.xlabel('Device')
plt.ylabel('Activity')
plt.tight_layout()

# Save the heatmap
accuracy_plot_filename = 'accuracy_heatmap.png'
plt.savefig(f'./plots/{accuracy_plot_filename}')
plt.close()
print(f"Accuracy Heatmap saved at: ./plots/{accuracy_plot_filename}")

# Plot Inference Time Heatmap
fig, ax = plt.subplots(figsize=(6, 6))
sns.heatmap(time_data, annot=True, cmap="YlGnBu", fmt=".5f", xticklabels=["CPU", "TPU"], yticklabels=labels)
ax.set_title("Heatmap: Per-Class Inference Time Comparison (CPU vs TPU)")
plt.xlabel('Device')
plt.ylabel('Activity')
plt.tight_layout()

# Save the heatmap
time_plot_filename = 'inference_time_heatmap.png'
plt.savefig(f'./plots/{time_plot_filename}')
plt.close()
print(f"Inference Time Heatmap saved at: ./plots/{time_plot_filename}")
