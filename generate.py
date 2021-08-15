from captcha.image import ImageCaptcha
import random
import os
from tqdm import trange

image = ImageCaptcha(fonts=['./fonts/font1.ttf'])

def randomSeqGenerator(captcha_len):
    ret = ""
    for i in range(captcha_len):
        num = chr(random.randint(48,57))#ASCII表示数字
        letter = chr(random.randint(97, 122))#取小写字母
        Letter = chr(random.randint(65, 90))#取大写字母
        s = str(random.choice([num,letter,Letter]))
        ret += s
    return ret
    
def captchaGenerator(dataset_path, dataset_len, captcha_len):
    os.makedirs(dataset_path, exist_ok=True)
    for i in trange(dataset_len):
        char_seq = randomSeqGenerator(captcha_len)
        save_path = os.path.join(dataset_path, f'{char_seq}.{i}.png')
        image.write(char_seq, save_path)

if __name__ == "__main__":
    captchaGenerator('./onechar/train', 100000, 1)
    captchaGenerator('./onechar/val', 20000, 1)
    captchaGenerator('./onechar/test', 20000, 1)