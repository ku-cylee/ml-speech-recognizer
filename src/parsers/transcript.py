import re

from config import TRANSCRIPT_PATH
from exceptions import FileParsingFailureException
from parsers import vectors

PTR = re.compile(r'"(trn/.+/.+/.+\.lab)"\n([\w\n]+\.)')

class Transcript:

    def __init__(self, match):
        self.filename = match.group(1).replace('.lab', '.txt')
        self.monophones = match.group(2).split('\n')

    
    def get_vectors(self):
        return vectors.parse(self.filename)


def parse():
    try:
        with open(TRANSCRIPT_PATH, 'r') as f:
            content = f.read()

        transcripts = [Transcript(match) for match in PTR.finditer(content)]
        transcripts[0].get_vectors()
        return transcripts
    except:
        raise FileParsingFailureException('Vector', file_path)
