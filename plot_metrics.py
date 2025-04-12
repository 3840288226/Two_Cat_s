import json
import matplotlib.pyplot as plt
import os

def plot_training_curves(train_losses, val_losses, val_mious, val_f1s):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(15, 5))

    # 第1个子图：Loss
    plt.subplot(1, 3, 1)
    plt.plot(epochs, train_losses, label='训练损失')
    plt.plot(epochs, val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.title('训练曲线')
    plt.legend()

    # 第2个子图：mIoU
    plt.subplot(1, 3, 2)
    plt.plot(epochs, val_mious, label='验证 mIoU')
    plt.xlabel('Epoch')
    plt.ylabel('mIoU')
    plt.title('验证 mIoU 变化')
    plt.legend()

    # 第3个子图：F1
    plt.subplot(1, 3, 3)
    plt.plot(epochs, val_f1s, label='验证 F1 得分')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('验证 F1 得分变化')
    plt.legend()

    plt.tight_layout()
    plt.savefig("results/training_metrics.png")
    print("图像已保存为：results/training_metrics.png")
    plt.show()

# === 关键部分：从 metrics.json 读取数据并调用函数 ===
if __name__ == '__main__':
    metrics_path = './results/metrics.json'
    if not os.path.exists(metrics_path):
        print("错误：未找到 metrics.json 文件！")
    else:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        train_losses = metrics.get('loss', [])
        val_losses = metrics.get('loss', [])  # 如果没有验证损失就先用训练损失
        val_mious = metrics.get('miou', [])
        val_f1s = metrics.get('f1', [])
        plot_training_curves(train_losses, val_losses, val_mious, val_f1s)
