import matplotlib.pyplot as plt
import pandas as pd

# Đường dẫn tới file log
log_file_path = "/Users/nguyendinhhuy/Documents/Master Course/Master Resource/UniCA-msc-ds_ai-main-1/semester3/CORE AI TRACK/federated_learning/Codes/TP1 26011920/log_files.txt"

# Đọc file log
data = pd.read_csv(
    log_file_path,
    sep="\t",
    header=None,
    names=["Train_Loss", "Train_Metric", "Test_Loss", "Test_Metric"]
)

# Tạo biểu đồ cho Loss
plt.figure(figsize=(10, 6))
plt.plot(data["Train_Loss"], label="Train Loss")
plt.plot(data["Test_Loss"], label="Test Loss")
plt.title("Training and Testing Loss")
plt.xlabel("Rounds")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.savefig("loss_curve.png")
plt.show()

# Tạo biểu đồ cho Metric
plt.figure(figsize=(10, 6))
plt.plot(data["Train_Metric"], label="Train Metric")
plt.plot(data["Test_Metric"], label="Test Metric")
plt.title("Training and Testing Metric")
plt.xlabel("Rounds")
plt.ylabel("Metric")
plt.legend()
plt.grid(True)
plt.savefig("/Users/nguyendinhhuy/Documents/Master Course/Master Resource/UniCA-msc-ds_ai-main-1/semester3/CORE AI TRACK/federated_learning/Codes/TP1 26011920/metric_curve.png")
plt.show()
