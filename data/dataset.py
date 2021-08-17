import torch
import torch.nn as nn
import torch.utils.data as data
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import os

from torchvision.transforms.transforms import Resize
from utils.config_util import configGetter

cfg = configGetter('DATASET')

HEIGHT = cfg['CAPTCHA']['IMG_HEIGHT']
WIDTH = cfg['CAPTCHA']['IMG_WIDTH']

class captcha_dataset(data.Dataset):
    def __init__(self, data_type: str) -> None:
        super().__init__()
        if data_type == 'train':
            self.data_path = cfg['TRAINING_DIR']
        elif data_type == 'val':
            self.data_path = cfg['TESTING_DIR']
        elif data_type == 'test':
            self.data_path = cfg['TESTING_DIR']
        else:
            raise ValueError('data_type must be train, val or test')
        self.data_list = os.listdir(self.data_path)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __getitem__(self, index):
        img_path = os.path.join(self.data_path, self.data_list[index])
        img = self.transform(Image.open(img_path))
        img = img.resize((HEIGHT, WIDTH))
        label = self.data_list[index].split('.')[0].lower()
        return img, str_to_vec(label)

    def __len__(self):
        return len(self.data_list)


def str_to_lst(s: str):
    lst = []
    for c in s:
        if '0' <= c <= '9':
            lst.append(ord(c) - ord('0'))
        elif 'a' <= c <= 'z':
            lst.append(ord(c) - ord('a') + 10)
        elif 'A' <= c <= 'Z':
            lst.append(ord(c) - ord('A') + 36)
    return lst


def lst_to_str(lst: list):
    s = ''
    for i in lst:
        if i < 10:
            s += chr(i + ord('0'))
        elif i < 36:
            s += chr(i + ord('a') - 10)
        else:
            s += chr(i + ord('A') - 36)
    return s


def str_to_onehotvec(s: str):
    return F.one_hot(torch.LongTensor(str_to_lst(s)), CLASS_NUM)


def str_to_vec(s: str):
    return torch.LongTensor(str_to_lst(s))


if __name__ == '__main__':
    d = captcha_dataset('./dataset', 'train')
    a = d[0]
    print(a[0].size(), a[1].size(), lst_to_str(a[1]))
