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


    def create_observation_table(self, vectors):
        for mixture in self.mixtures:
            mixture.create_observation_table(vectors)

        self.observ_probs = []
        for time in range(len(vectors)):
            probs = [mix.observ_probs[time] for mix in self.mixtures]
            self.observ_probs.append(lib.sum_logs(probs))


class Mixture:

    def __init__(self, text_data):
        is_mean_line = False
        is_variance_line = False
        for idx, line in enumerate(text_data.split('\n')):
            if idx == 0:
                self.weight = lib.refined_log(float(line.split(' ')[-1]))

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


    def create_observation_table(self, vectors):
        self.observ_probs = []

        gaussian_constant = 0.5 * self.dimension * math.log(2 * math.pi)
        variances_sum = 0.5 * sum(lib.refined_log(gauss.variance) for gauss in self.gaussians)
        constant = self.weight - gaussian_constant - variances_sum

        for vector in vectors:
            differences_sum = 0
            for idx, gaussian in enumerate(self.gaussians):
                differences_sum += ((vector.values[idx] - gaussian.mean) / gaussian.variance) ** 2
            self.observ_probs.append(constant - 0.5 * differences_sum)


class Gaussian:

    def __init__(self, mean, variance):
        self.mean = mean
        self.variance = variance


    def add_mean(self, log_value):
        self.mean += math.exp(log_value)


    def add_variance(self, log_value):
        self.mean += math.exp(log_value)


class TransitionTable:

    def __init__(self, text_data):
        rows = text_data.split('\n')[1:-1]
        self.probabilities = [[lib.refined_log(float(prob)) for prob in row.strip().split(' ')] for row in rows]


def parse():
    with open(HMM_PATH, 'r') as f:
        content = f.read()

    pheno_models_list = [PhenomenonModel(text.strip()) for text in content.split('~h')[1:]]
    pheno_models = dict()
    for pheno in pheno_models_list:
        pheno_models[pheno.phenomenon] = pheno
    return pheno_models
