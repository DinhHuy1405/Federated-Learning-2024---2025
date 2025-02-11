import os
import matplotlib.pyplot as plt
import pandas as pd

# Define the path to the result file
file_path = "/Users/nguyendinhhuy/Documents/Master Course/Master Resource/UniCA-msc-ds_ai-main-1/semester3/CORE AI TRACK/federated_learning/Codes/TP1 Ex1,2,3/result.txt"

if not os.path.exists(file_path):
    print(f"File {file_path} not found. Please run the training script to generate the result.txt file.")
    exit()

data = []

with open(file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(', ')
        if len(parts) >= 3:
            local_steps = int(parts[0].split(": ")[1].strip())
            round_number = int(parts[1].split(": ")[1].strip())
            test_accuracy = float(parts[2].split(": ")[1].strip())
            data.append((local_steps, round_number, test_accuracy))

# Create DataFrame
df = pd.DataFrame(data, columns=["Local Steps", "Round", "Test Accuracy"])

# Handle duplicates before pivoting
df = df.drop_duplicates(subset=["Round", "Local Steps"], keep='last')

# Plot test accuracy vs. number of training rounds
plt.figure(figsize=(10, 6))
local_steps_list = [1, 5, 10, 50, 100]
for local_steps in local_steps_list:
    subset = df[df["Local Steps"] == local_steps]
    plt.plot(subset["Round"], subset["Test Accuracy"], marker='o', label=f'Local Steps: {local_steps}')
plt.xlabel("Number of Training Rounds", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Impact of Local Epochs on Test Accuracy (Rounds)", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("/Users/nguyendinhhuy/Documents/Master Course/Master Resource/UniCA-msc-ds_ai-main-1/semester3/CORE AI TRACK/federated_learning/Codes/TP1 Ex1,2,3/accuracy_plot_rounds.png", dpi=300, bbox_inches='tight')
print("Saved plot: accuracy_plot_rounds.png")
plt.show()

# Plot test accuracy vs. number of local epochs
plt.figure(figsize=(10, 6))
for local_steps in local_steps_list:
    subset = df[df["Local Steps"] == local_steps]
    plt.plot(subset["Local Steps"], subset["Test Accuracy"], marker='o', label=f'Local Steps: {local_steps}')
plt.xlabel("Number of Local Epochs", fontsize=12)
plt.ylabel("Test Accuracy", fontsize=12)
plt.title("Impact of Local Epochs on Test Accuracy (Epochs)", fontsize=14)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.savefig("/Users/nguyendinhhuy/Documents/Master Course/Master Resource/UniCA-msc-ds_ai-main-1/semester3/CORE AI TRACK/federated_learning/Codes/TP1 Ex1,2,3/accuracy_plot_epochs.png", dpi=300, bbox_inches='tight')
print("Saved plot: accuracy_plot_epochs.png")
plt.show()
