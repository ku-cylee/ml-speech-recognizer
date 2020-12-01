import os

from dotenv import load_dotenv

load_dotenv(verbose=True)

HMM_PATH = os.getenv('HMM_PATH', './data/hmm.txt')
MONOPHONE_PATH = os.getenv('MONOPHONE_PATH', './data/trn_mono.txt')
TRAIN_DATA_PATH = os.getenv('TRAIN_DATA_PATH', './data/trn/')
