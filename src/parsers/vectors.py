from exceptions import FileParsingFailureException

class Vector:
    
    def __init__(self, text_data):
        self.values = [float(val) for val in text_data.replace('  ', ' ').split(' ')]


def parse(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()

        vectors = [Vector(text.strip()) for text in content.split('\n')[1:]]
        return vectors
    except:
        raise FileParsingFailureException('Vector', file_path)
