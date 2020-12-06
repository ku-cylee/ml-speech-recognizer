from config import HMM_PATH
from exceptions import FileParsingFailureException

class PhenomenonModel:

    def __init__(self, text_data):
        self.phenomenon = text_data.split('"')[1]
        state_splitted = text_data.split('<STATE>')
        text_splitted = state_splitted[1:-1] + state_splitted[-1].split('<TRANSP>')
        self.states = [State(text.strip()) for text in text_splitted[:-1]]
        self.transition_table = TransitionTable(text_splitted[-1].strip())
        

class State:

    def __init__(self, text_data):
        self.id = int(text_data.split('\n')[0])
        self.mixtures = [Mixture(text.strip()) for text in text_data.split('<MIXTURE>')[1:]]


class Mixture:

    def __init__(self, text_data):
        is_mean_line = False
        is_variance_line = False
        for idx, line in enumerate(text_data.split('\n')):
            if idx == 0:
                self.weight = float(line.split(' ')[-1])

            if is_mean_line:
                means = [float(mean) for mean in line.strip().split(' ')]
                is_mean_line = False

            if is_variance_line:
                variances = [float(variance) for variance in line.strip().split(' ')]
                is_variance_line = False

            is_mean_line = line.startswith('<MEAN>')
            is_variance_line = line.startswith('<VARIANCE>')

        self.dimension = len(means)
        self.gaussians = [Gaussian(means[i], variances[i]) for i in range(self.dimension)]


class Gaussian:

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


class TransitionTable:

    def __init__(self, text_data):
        self.probabilities = [[float(prob) for prob in row.strip().split(' ')] for row in text_data.split('\n')[1:-1]]


def parse():
    try:
        with open(HMM_PATH, 'r') as f:
            content = f.read()

        pheno_models = [PhenomenonModel(text.strip()) for text in content.split('~h')[1:]]
        return pheno_models
    except:
        raise FileParsingFailureException('HMM Model', HMM_PATH)
