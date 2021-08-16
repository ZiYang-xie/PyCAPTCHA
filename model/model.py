import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from data.dataset import HEIGHT, WIDTH, CLASS_NUM, CHAR_LEN, lst_to_str


def eval_acc(label, pred):
    # label: CHAR_LEN x batchsize
    # pred: CHAR_LEN x batchsize x CLASS_NUM
    pred_res = pred.argmax(dim=2) # CHAR_LEN x batchsize
    eq = ((pred_res == label).float().sum(dim=0)==CHAR_LEN).float() #batchsize
    return eq.sum()/eq.size(0)

class captcha_model(pl.LightningModule):
    def __init__(self, model, lr=1e-4, optimizer=None):
        super(captcha_model, self).__init__()
        self.model = model
        self.lr = lr
        self.optimizer = optimizer

    def forward(self, x):
        x = self.model(x)
        return x

    def step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).permute(1, 0, 2)
        y = y.permute(1, 0)
        loss = 0
        for i in range(CHAR_LEN):
            loss += F.cross_entropy(y_hat[i], y[i])
        return loss, y, y_hat

    def training_step(self, batch, batch_idx):
        loss, label, y = self.step(batch, batch_idx)
        self.log("train loss", loss.item())
        self.log("train acc", eval_acc(label, y))
        return loss

    def validation_step(self, batch, batch_idx):
        loss, label, y = self.step(batch, batch_idx)
        self.log("val loss", loss.item())
        self.log("val acc", eval_acc(label, y))
        return loss

    def test_step(self, batch, batch_idx):
        loss, label, y = self.step(batch, batch_idx)
        self.log("test loss", loss.item())
        self.log("test acc", eval_acc(label, y))
        if batch_idx == 0:
            label = label.permute(1, 0)
            y = y.permute(1, 0, 2)
            pred = y.argmax(dim=2)
            res = [f"pred:{lst_to_str(pred[i])}, real:{lst_to_str(label[i])}" for i in range(pred.size(0))]
            print("\n".join(res))
        return loss

    def configure_optimizers(self):
        if self.optimizer is None:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        else:
            optimizer = self.optimizer
        return optimizer


class model_resnet(torch.nn.Module):
    def __init__(self):
        super(model_resnet, self).__init__()
        self.resnet = models.resnet18(pretrained=False)
        self.resnet.fc = nn.Linear(512, CHAR_LEN*CLASS_NUM)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), CHAR_LEN, CLASS_NUM)
        return x


class model_conv(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # 3x160x60
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 32x80x30
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 64x40x15
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 128x20x7
        self.fc = nn.Sequential(
            nn.Linear(128*(WIDTH//8)*(HEIGHT//8), 1024),
            nn.ReLU(),
            nn.Linear(1024, CLASS_NUM*CHAR_LEN)
        )
        # CLASS_NUM*4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), CHAR_LEN, CLASS_NUM)
        return x
