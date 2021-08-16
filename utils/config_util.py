import yaml
import os

def configGetter(PARAM):
    with open("./config/config.yaml", 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
        return cfg if PARAM is None else cfg[PARAM]
