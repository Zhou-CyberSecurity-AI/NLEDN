# ======================= Pytorch Lib =============================
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
# ======================= My Lib ===================================
from model.Main_Block import Main_Block
from DataLodaer.Trainloader import DataSet
from utils.utils import calc_psnr, calc_ssim
# ======================= Config File ===============================
import Config.config as cfg
# ======================= Origin Lib ================================
import os
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train(**kwargs):
    dataset = kwargs['dataloader']
    model = kwargs['model']
    num_epochs = kwargs['num_epochs']
    savepath = kwargs['savepath']
    parameters = list(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=1e-3, weight_decay=1e-5)
    criterion = nn.L1Loss
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_index, (img_batch, label_batch, name_list) in enumerate(dataset.train_loader):
            print('[%d/%d]' % (batch_index, train_batches), name_list[0])

            # ======================= check gpu availability ================================
            if torch.cuda.is_available():
                img_batch = img_batch.to(device)
                label_batch = label_batch.to(device)
            label_res_batch = img_batch - label_batch
            # ======================= The weight parameter gradient is cleared to 0 ================================
            optimizer.zero_grad()

            # ======================= Forward and back propagation ================================
            prediction_res_batch = net(img_batch)
            prediction_batch = img_batch - prediction_res_batch
            prediction_batch = torch.clamp(prediction_batch, 0, 1)

            loss = criterion(prediction_batch, label_res_batch)
            loss.backward()
            optimizer.step()

            # ======================= view loss value ================================
            running_loss += loss.item()
            if batch_index % 10 == 9:
                print('[%5d, %5d] loss: %.3f' % (epoch + 1, batch_index + 1, running_loss / 10))
                running_loss = 0.0
    torch.save(model.state_dict(), savepath)


if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    import argparse

    parser = argparse.ArgumentParser(description='train',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--Epochs', default=200, type=int)
    parser.add_argument('--save_dir', default='../weights')
    args = parser.parse_args()
    print(args)
    # ======================= Config ===================================
    print('-' * 40)
    print('cuda number:', cfg.CUDA_NUMBER, '\n')
    print('train dir:', cfg.train_dir)

    # ======================= DataSet ===================================
    dataset = DataSet(cfg)
    train_batches = dataset.train_loader.__len__()
    train_samples = dataset.train_dataset.__len__()

    print('Train: %d batches, %d samples' % (train_batches, train_samples))
    print('-' * 40 + '\n')

    # ==================== Network ======================
    net = Main_Block().to(device)

    # ==================== output network struct ======================
    from torchsummary import summary

    summary(net, input_size=(3, 481, 321))
    exit(0)

    # ================== Network to GPU =========================
    if torch.cuda.is_available():
        net.cuda(cfg.CUDA_NUMBER)

    # ================== Train Network =========================
    train(dataloader=dataset, model=net, num_epochs=args.Epochs, savepath=cfg.weight_path)
