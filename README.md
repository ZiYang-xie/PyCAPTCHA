# PyCAPTCHA 🔍
![版本号](https://img.shields.io/badge/Version-Beta--0.0.1-blue) ![作者](https://img.shields.io/badge/Author-Xzy-orange)  

![](./assets/captcha.png)
---

**An End-to-end Pytorch-Lightning implemented CAPTCHA OCR model.**  
Training 2 epoch under 100k images to get over 96% acc on Val dataset 🤩  
> with 200k or even more training set you may get >98% acc

![](./assets/testing.png)



## INSTALL ⚙️
### Step1: Create & Activate Conda Env
```shell
conda create -n "PyCaptcha" python=3.7
conda activate PyCaptcha
```

### Step2: Install Pytorch 
```shell
conda install torch==1.9.0 torchvision==0.10.0 -c pytorch
```

### Step3: Install Pip Requirements 
```shell
pip install -r requirement.txt
```

## Training 🚀
### Step1: Set the Config file
Check out the yaml file  
```yaml
DATASET:
  DATASET_DIR: './dataset'
  TRAINING_DIR: './dataset/train'
  TESTING_DIR: './dataset/val'
...

SOLVER:
  LR: 5.0e-4
  BATCH_SIZE: 256

LOGGER:
  CHECKPOINT_DIR: './checkpoint'
  LOG_DIR: './logs'
```


### Step2: Check the Dataset
Make sure you have a dataset, you can generate the dataset with the ```utils/captcha_generater.py``` script
```shell
python utils/captcha_generater.py
```

### Step3: Start Training
```shell
python launcher.py --exp_name "my_exp" #Start Training
```
*check out the ```utils/arg_parsers.py``` for details*

> The tensorboard and ckpt file will save at ```logs```

## Testing 📝
```shell
python test.py --ckpt "your_ckpt"  #Start Testing
```
*check out the ```utils/arg_parsers.py``` for details*
