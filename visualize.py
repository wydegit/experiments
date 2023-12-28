import matplotlib.pyplot as plt
import re
import os
import numpy as np
def train_visualize(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    epoch = []
    loss = []
    pattern = r"Epoch:(\d+)  Training loss:([\d.]+)  learning rate:([\d.]+)"
    with open(path, "r") as f:
        for line in f.readlines():
            match = re.search(pattern, line)
            if match:
                epoch.append(int(match.group(1)))
                loss.append(float(match.group(2)))
            else:
                pass

    fig, ax = plt.subplots()
    # plt.rc('font', family='Times New Roman')
    plt.rc('mathtext', fontset='stix')
    fig.set_size_inches(8, 6)

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training loss")

    ax.plot(epoch, loss, color='b', linestyle='-', marker='.', label='Training loss')

    ax.set_title("Training loss curve")
    # ax.legend()
    visual_name = os.path.basename(path).split('.')[0]
    fig.savefig(os.path.join(save_path, visual_name + '.png'))
    fig.show()

def val_visualize(path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    epoch = []
    loss = []
    miou = []
    niou = []
    precision = []
    recall = []
    tpr = []
    fpr = []
    pattern = r"Epoch:(\d+)  val_loss:([\d.]+)  mIoU:([\d.]+)  nIoU:([\d.]+)"
    # pattern = r"Epoch:(\d+)  val_loss:([\d.]+)  mIoU:([\d.]+)  nIoU:([\d.]+)  precision:[]  recall:\[([\d. ]+)\]"

    with open(path, "r") as f:
        for line in f.readlines():
            match = re.search(pattern, line)
            if match:
                epoch.append(int(match.group(1)))
                loss.append(float(match.group(2)))
                miou.append(float(match.group(3)))
                niou.append(float(match.group(4)))
                # precision.append(float(match.group(6)))
                # recall.append(float(match.group(7)))
                # tpr.append(float(match.group(8)))
                # fpr.append(float(match.group(9)))
            else:
                pass


    fig, ax = plt.subplots()
    fig.suptitle("Validation process")
    fig.set_size_inches(8, 6)
    # plt.rc('font', family='Times New Roman')
    plt.rc('mathtext', fontset='stix')


    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation metrics")


    ax.plot(epoch, loss, color='r', linestyle='-', marker='.', label='Validation loss')
    ax.plot(epoch, miou, color='g', linestyle='-', marker='.', label='mIoU')
    ax.plot(epoch, niou, color='b', linestyle='-', marker='.', label='nIoU')
    fig.legend(bbox_to_anchor=(1.04, 1), loc="upper right")


    visual_name = os.path.basename(path).split('.')[0]
    plt.savefig(os.path.join(save_path, visual_name + '.png'))
    plt.show()


def plot_img_and_mask(img, label, output, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig, ax = plt.subplots(1, 3)
    fig.set_size_inches(8, 6)
    plt.rc('font', family='Times New Roman')
    plt.rc('mathtext', fontset='stix')

    ax[0].imshow(img, cmap='gray')
    ax[0].set_title("Input image")

    ax[1].imshow(label, cmap='gray')
    ax[1].set_title("Ground truth")

    ax[2].imshow(output, cmap='gray')
    ax[2].set_title("Prediction")

    for a in ax:
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'predict.png'))
    plt.show()




if __name__ == "__main__":
    dir = r'D:/wang/results/test/'
    train_name = 'ACMFPN-straight_open-sirst-v2__train_log.txt'
    val_name = 'ACMFPN-straight_open-sirst-v2__val_log.txt'

    train_visualize(os.path.join(dir + train_name), dir)
    val_visualize(os.path.join(dir + val_name), dir)