import glob
import numpy as np
import os
import scipy.io as scio
import torch
from torch.utils.data import Dataset

class trainset_loader(Dataset):
    def __init__(self, root):
        self.file_path = 'input'
        self.files_A = sorted(glob.glob(os.path.join(root, 'train', self.file_path, 'data') + '*.mat'))
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_B = file_A.replace(self.file_path,'label')
        file_C = file_A.replace('input','projection')
        file_D = file_A.replace('input','geometry')
        input_data = scio.loadmat(file_A)['data']
        label_data = scio.loadmat(file_B)['data']
        prj_data = scio.loadmat(file_C)['data']
        geometry = scio.loadmat(file_D)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        label_data = torch.FloatTensor(label_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data)
        geometry = torch.FloatTensor(geometry).view(-1)
        option = geometry[:-1]
        idx = torch.tensor([0,1,4,5,7,8,10])
        feature = geometry[idx]
        feature[0] = torch.log2(feature[0])
        feature[1] = feature[1] / 256
        feature[6] = torch.log10(feature[6])
        minVal = torch.FloatTensor([5,0,0.005-0.001,0.004-0.0015,2.0-0.5,2.0-0.5,3.5])
        maxVal = torch.FloatTensor([11,4,0.012+0.001,0.014+0.0015,5.0+0.5,5.0+0.5,6.5])
        feature = (feature - minVal) / (maxVal - minVal)
        return input_data, label_data, prj_data, option, feature

    def __len__(self):
        return len(self.files_A)

class testset_loader(Dataset):
    def __init__(self, root):
        self.files_A = []
        for i in range(0,5):
            root_path = root + '_' + str(i+1)
            path = os.path.join(root_path, 'test', 'input', 'data')
            self.files_A = self.files_A + sorted(glob.glob(path + '*.mat'))
        self.gemoetry = torch.FloatTensor(
            [
                [1024, 512, 256, 256, 0.006641, 0.0072, 0.006134, 2.5, 2.5, 0, 1e5],
                [88, 768, 256, 256, 0.0078125, 0.0058, 0.0714, 3.5, 3.0, 0, 1e6],
                [1024, 768, 256, 256, 0.01, 0.0062, 0.006134, 5.0, 4.0, 0, 5e4],
                [128, 512, 256, 256, 0.012, 0.014, 0.0491, 5.0, 5.0, 0, 2.5e5],
                [108, 512, 256, 256, 0.005, 0.004, 0.0582, 4.0, 2.0, 0, 5e5],
                [1024, 512, 256, 256, 0.0060, 0.0060, 0.006134, 3.0, 3.0, 0, 8e4],
                [1024, 768, 256, 256, 0.008, 0.0056, 0.006134, 5.0, 5.0, 0, 7e4],
                [144, 512, 256, 256, 0.01, 0.011, 0.0436, 5.0, 5.0, 0, 3e5],
                [128, 512, 256, 256, 0.007, 0.0075, 0.0491, 4.0, 3.0, 0, 5e5]
            ]
        )
        
    def __getitem__(self, index):
        file_A = self.files_A[index]
        file_C = file_A.replace('input','projection')
        geometry_idx = int(file_A[47])
        res_name = 'result/geometry_' + str(geometry_idx) + '/'
        if not os.path.exists(res_name):
            os.makedirs(res_name)
        res_name = res_name + file_A[-13:]
        input_data = scio.loadmat(file_A)['data']
        prj_data = scio.loadmat(file_C)['data']
        input_data = torch.FloatTensor(input_data).unsqueeze_(0)
        prj_data = torch.FloatTensor(prj_data)
        geometry = self.gemoetry[geometry_idx-1]
        geometry = torch.FloatTensor(geometry).view(-1)
        option = geometry[:-1]
        idx = torch.tensor([0,1,4,5,7,8,10])
        feature = geometry[idx]
        feature[0] = torch.log2(feature[0])
        feature[1] = feature[1] / 256
        feature[6] = torch.log10(feature[6])
        minVal = torch.FloatTensor([5,0,0.005-0.001,0.004-0.0015,2.0-0.5,2.0-0.5,3.5])
        maxVal = torch.FloatTensor([11,4,0.012+0.001,0.014+0.0015,5.0+0.5,5.0+0.5,6.5])
        feature = (feature - minVal) / (maxVal - minVal)
        return input_data, prj_data, res_name, option, feature

    def __len__(self):
        return len(self.files_A)
