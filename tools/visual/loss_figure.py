import re
import matplotlib.pyplot as plt
import os

def visual_loss(file_path_list, save_path_list):


    # 正则表达式匹配模式
    pattern = re.compile(r"\[.*?INFO: Step: (\d+)/\d+.*?Loss: ([\d.]+)")

    for file_path, save_path in zip(file_path_list, save_path_list):
        # 用于存储步骤数和损失值的列表
        steps = []
        losses = []
        # 读取文件并解析每一行
        with open(file_path, 'r') as file:
            for line in file:
                match = pattern.search(line)
                if match:
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    steps.append(step)
                    losses.append(loss)

        # 打印解析出的步数和损失值（可选）
        # print("Steps: ", steps)
        # print("Losses: ", losses)

        # 创建绘图
        plt.figure(figsize=(30, 18))
        plt.plot(steps, losses, marker='o', linestyle='-', color='b')

        # 设置标题和标签
        plt.title('Training Loss Over Steps')

        plt.xlabel('Step', fontsize=20)
        plt.ylabel('Loss', fontsize=30)

        # 设置刻度标签字体大小
        plt.xticks(fontsize=25)
        plt.yticks(fontsize=25)

        # 显示网格
        plt.grid(True)
        # 保存图表到文件
        plt.savefig(os.path.join(save_path, f'loss_{file_path[2:-4]}_high.png'))
        # 显示图表
        plt.clf()


if __name__ == '__main__':
    save_path_list = ["./outputs/loss",]
    file_path_list = ['./UniAnimate_log_tmux.txt',]
    visual_loss(file_path_list, save_path_list)