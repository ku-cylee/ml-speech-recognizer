import math

import lib

class ParamsProcessor:

    def __init__(self, models, state_occupancy_table, transcripts_count):
        self.models = models.values()
        self.state_occupancy_table = state_occupancy_table
        self.transcripts_count = lib.refined_log(transcripts_count)


    def process(self):
        for model in self.models:
            self.apply_trans_table_per_model(model)
            self.apply_gaussians(model)

    
    def apply_trans_table_per_model(self, model):
        trans_table = model.transition_table.probabilities
        new_table = model.new_table
        states_count = len(model.states)
        for sidx in range(states_count):
            trans_table[0][sidx + 1] = new_table[0][sidx + 1] - self.transcripts_count

        for psidx in range(states_count):
            for nsidx in range(states_count):
                state_occ = self.state_occupancy_table[psidx][0]
                trans_table[psidx + 1][nsidx + 1] = new_table[psidx + 1][nsidx + 1] - state_occ
            
            trans_table[psidx + 1][-1] = lib.get_remaining_prob(trans_table[psidx + 1][1:-1])


    def apply_gaussians(self, model):
        for sidx in enumerate(model.states):
            for midx in range(len(model.states[sidx].mixtures)):
                self.apply_gaussians_per_mixture(model, sidx, midx)

    
    def apply_gaussians_per_mixture(self, model, sidx, midx):
        state = model.states[sidx]
        mixture = state.mixtures[midx]

        mixture_state_occ = self.state_occupancy_table[sidx][midx + 1]
        init_state_occ = self.state_occupancy_table[sidx][0]
        mixture.weight = mixture_state_occ - init_state_occ

        exp_state_occ = math.exp(mixture_state_occ)
        for gaussian in mixture.gaussians:
            new_mean = gaussian.new_mean
            gaussian.mean = new_mean / exp_state_occ
            gaussian.variance = gaussian.new_variance / exp_state_occ - new_mean ** 2
