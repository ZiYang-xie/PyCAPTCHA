from model.model import captcha_model, model_conv, model_resnet
from datamodule import captcha_dm
import pytorch_lightning as pl
import torch.optim as optim
import torch
from utils.config_util import configGetter

if __name__ == "__main__":
    cfg  = configGetter('SOLVER')
    lr = cfg['LR']
    batch_size = cfg['BATCH_SIZE']

    pl.seed_everything(42)

    m = model_resnet()
    model = captcha_model(
        model=m, lr=lr)
    dm = captcha_dm(batch_size=batch_size)

    tb_logger = pl.loggers.TensorBoardLogger(
        "./logs/", name='main', version=2, default_hp_metric=False)
        
    trainer = pl.Trainer(deterministic∏=True,
                         gpus=-1,
                         auto_select_gpus=True,
                         precision=32,
                         logger=tb_logger,
                         fast_dev_run=False,
                         max_epochs=5,
                         log_every_n_steps=50,
                         stochastic_weight_avg=True
                         )
    trainer.fit(model, datamodule=dm)
    torch.save(model.state_dict(), "./checkpoint/model.pth")
