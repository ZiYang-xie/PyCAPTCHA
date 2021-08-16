from model.model import captcha_model, model_conv, model_resnet
from data.datamodule import captcha_dm
from utils.arg_parsers import test_arg_parser
import pytorch_lightning as pl

def test(args):
    dm = captcha_dm()
    model = captcha_model.load_from_checkpoint(args.ckpt, model=model_resnet())
    tb_logger = pl.loggers.TensorBoardLogger(
        args.log_dir, name=args.test_name, version=2, default_hp_metric=False)
    trainer = pl.Trainer(deterministic=True,
                         gpus=-1,
                         auto_select_gpus=True,
                         precision=32,
                         logger=tb_logger,
                         fast_dev_run=False,
                         max_epochs=5,
                         log_every_n_steps=50,
                         stochastic_weight_avg=True
                         )
    trainer.test(model, dm)

if __name__ == "__main__":
    args = test_arg_parser()
    test(args)
