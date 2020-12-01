import argparse
import os
import re
import glob
import numpy as np
import scipy.io as sio
from vis_tools import Visualizer

import torch
import torch.nn as nn
import torch.optim as optim
import model

from datasets import trainset_loader
from datasets import testset_loader
from torch.utils.data import DataLoader
from torch.autograd import Variable
from skimage.measure import compare_psnr
from skimage.measure import compare_ssim
import time

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-4, help="adam: learning rate")
parser.add_argument("--n_block", type=int, default=50)
parser.add_argument("--n_cpu", type=int, default=4)
parser.add_argument("--model_save_path", type=str, default="saved_models/1st")
parser.add_argument('--checkpoint_interval', type=int, default=1)

opt = parser.parse_args()
cuda = True if torch.cuda.is_available() else False
train_vis = Visualizer(env='training_learn')

def my_collate_test(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    prj_data = [item[1] for item in batch]
    res_name = [item[2] for item in batch]
    option = torch.stack([item[3] for item in batch], 0)
    feature = torch.stack([item[4] for item in batch], 0)
    return input_data, prj_data, res_name, option, feature

def my_collate(batch):
    input_data = torch.stack([item[0] for item in batch], 0)
    label_data = torch.stack([item[1] for item in batch], 0)
    prj_data = [item[2] for item in batch]
    option = torch.stack([item[3] for item in batch], 0)
    feature = torch.stack([item[4] for item in batch], 0)
    return input_data, label_data, prj_data, option, feature

class net():
    def __init__(self):
        self.model = model.Learn(opt.n_block)
        self.loss = nn.MSELoss()
        self.path = opt.model_save_path
        self.train_data = DataLoader(trainset_loader("E:\\StudyData\\原始数据\\meta_learning"),
            batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu, collate_fn=my_collate)
        self.test_data = DataLoader(testset_loader("E:\\StudyData\\原始数据\\different gemotries\\geometry"),
            batch_size=opt.batch_size*4, shuffle=False, num_workers=opt.n_cpu, collate_fn=my_collate_test)
        self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, weight_decay=1e-8)
        self.start = 0
        self.epoch = opt.epochs
        self.check_saved_model()
        if cuda:
            self.model = self.model.cuda()

    def check_saved_model(self):
        if not os.path.exists(self.path):
            os.makedirs(self.path)
            self.initialize_weights()
        else:
            model_list = glob.glob(self.path + '/model_epoch_*.pth')
            if len(model_list) == 0:
                self.initialize_weights()
            else:
                last_epoch = 0
                for model in model_list:
                    epoch_num = int(re.findall(r'model_epoch_(-?[0-9]\d*).pth', model)[0])
                    if epoch_num > last_epoch:
                        last_epoch = epoch_num
                self.start = last_epoch
                self.model.load_state_dict(torch.load(
                    '%s/model_epoch_%04d.pth' % (self.path, last_epoch), map_location='cpu'))

    def displaywin(self, img, low=0.42, high=0.62):
        img[img<low] = low
        img[img>high] = high
        img = (img - low)/(high - low) * 255
        return img

    def initialize_weights(self):
        for module in self.model.modules():
            if isinstance(module, model.prj_module):
                nn.init.normal_(module.weight, mean=0.02, std=0.001)
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()

    def train(self):
        self.model.train(mode=True)
        for epoch in range(self.start, self.epoch):
            for batch_index, data in enumerate(self.train_data):
                input_data, label_data, prj_data, options, feature_vec = data
                temp = []
                if cuda:
                    input_data = input_data.cuda()
                    label_data = label_data.cuda()
                    options = options.cuda()
                    feature_vec = feature_vec.cuda()
                    for i in range(len(prj_data)):
                        temp.append(torch.FloatTensor(prj_data[i]).cuda())
                    prj_data = temp                    
                self.optimizer.zero_grad()
                output = self.model(input_data, prj_data, options, feature_vec)
                loss = self.loss(output, label_data)
                loss.backward()
                self.optimizer.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d]: [loss: %f]"
                    % (epoch+1, self.epoch, batch_index+1, len(self.train_data), loss.item())
                )                
                train_vis.plot('Loss', loss.item())
                train_vis.img('Ground Truth', self.displaywin(label_data.detach()).cpu())
                train_vis.img('Result', self.displaywin(output.detach()).cpu())
                train_vis.img('Input', self.displaywin(input_data.detach()).cpu())
            if opt.checkpoint_interval != -1 and (epoch+1) % opt.checkpoint_interval == 0:
                torch.save(self.model.state_dict(), '%s/model_epoch_%04d.pth' % (self.path, epoch+1))

    def test(self):
        self.model.eval()
        for batch_index, data in enumerate(self.test_data):
            input_data, prj_data, res_name, options, feature_vec = data
            temp = []
            if cuda:
                input_data = input_data.cuda()
                options = options.cuda()
                feature_vec = feature_vec.cuda()
                for i in range(len(prj_data)):
                    temp.append(torch.FloatTensor(prj_data[i]).cuda())
                prj_data = temp                    
            with torch.no_grad():
                output = self.model(input_data, prj_data, options, feature_vec)
            res = output.cpu().numpy()
            for i in range(output.shape[0]):
                sio.savemat(res_name[i], {'data':res[i,0]})

if __name__ == "__main__":
    network = net()
    network.train()
    network.test()