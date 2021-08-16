from model import captcha_model, model_conv, model_resnet
from datamodule import captcha_dm
import pytorch_lightning as pl

CHKPATH = 'logs/main/version_1/checkpoints/epoch=1-step=1539.ckpt'

if __name__ == "__main__":
    dm = captcha_dm()
    model = captcha_model.load_from_checkpoint(CHKPATH,model=model_resnet())
    tb_logger = pl.loggers.TensorBoardLogger(
        "./logs/", name='main', version=2, default_hp_metric=False)
    trainer = pl.Trainer(deterministic=True,
                         gpus=-1,
                         auto_select_gpus=True,
                         precision=32,
                         logger=tb_logger,
                         fast_dev_run=False,
                         max_epochs=2,
                         log_every_n_steps=50,
                         stochastic_weight_avg=True
                         )
    trainer.test(model, dm)
