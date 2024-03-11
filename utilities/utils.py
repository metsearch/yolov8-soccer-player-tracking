import os
from glob import glob
from .log import logger

def pull_files(path2files, extension='jpg'):
    return glob(f'{path2files}/*.{extension}')

if __name__ == '__main__':
    logger.info('Testing utils...')