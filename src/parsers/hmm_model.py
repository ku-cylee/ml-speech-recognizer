import math

import lib

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


    def get_observation_prob(self, x):
        return lib.sum_logs(mixture.get_indiv_observation_prob(x) for mixture in self.mixtures)


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


    def get_indiv_observation_prob(self, vector):
        log_weight = math.log(self.weight)
        constant = 0.5 * self.dimension * math.log(2 * math.pi)
        variances_sum = sum(math.log(gaussian.variance) for gaussian in self.gaussians)

        differences_sum = 0
        for i in range(self.dimension):
            gaussian = self.gaussians[i]
            differences_sum = ((vector[i] - gaussian.mean) / gaussian.variance) ** 2
            
        return log_weight - constant - variances_sum - 0.5 * differences_sum


class Gaussian:

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance

class TransitionTable:

    def __init__(self, text_data):
        self.probabilities = [[float(prob) for prob in row.strip().split(' ')] for row in text_data.split('\n')[1:-1]]


def parse():
    with open(HMM_PATH, 'r') as f:
        content = f.read()

    pheno_models_list = [PhenomenonModel(text.strip()) for text in content.split('~h')[1:]]
    pheno_models = dict()
    for pheno in pheno_models_list:
        pheno_models[pheno.phenomenon] = pheno
    return pheno_models
