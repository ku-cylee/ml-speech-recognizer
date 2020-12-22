import os

from config import TRAIN_DATA_PATH

class Vector:
    
    def __init__(self, text_data):
        self.values = [float(val) for val in text_data.replace('  ', ' ').split(' ')]


def parse(filename):
    file_path = os.path.join(TRAIN_DATA_PATH, filename)
    with open(file_path, 'r') as f:
        content = f.read().strip()

    vectors = [Vector(text.strip()) for text in content.split('\n')[1:]]
    return vectors
