import pytorch_lightning as pl
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
import torch
from dataset import HEIGHT, WIDTH, CHARNUM, CHARLEN


def eval_acc(label, pred):
    acc = 0
    for i in range(CHARLEN):
        acc += (label[i] == torch.argmax(pred[i], dim=1)
                ).sum().item()/label[i].shape[0]
    return acc/len(label)


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
        loss = torch.tensor([F.cross_entropy(y_hat[i], y[i])
                            for i in range(CHARLEN)], requires_grad=True).sum()
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
        self.resnet.fc = nn.Linear(512, CHARLEN*CHARNUM)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), CHARLEN, CHARNUM)
        return x

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
        for i in range(CHARLEN):
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
        self.resnet.fc = nn.Linear(512, CHARLEN*CHARNUM)

    def forward(self, x):
        x = self.resnet(x)
        x = x.view(x.size(0), CHARLEN, CHARNUM)
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
            nn.Linear(1024, CHARNUM*CHARLEN)
        )
        # CHARNUM*4

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), CHARLEN, CHARNUM)
        return x