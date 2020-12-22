import math

import lib

class TranscriptProcessor:

    def __init__(self, models):
        self.models = models
        for model in self.models.values():
            model.create_new_transition_table()
            for state in model.states:
                for mixture in state.mixtures:
                    for gaussian in mixture.gaussians:
                        gaussian.set_new_params()


    def process(self, transcript):
        self.init_per_transcript(transcript)
        self.calculate_parameters()
        self.apply_changed_values()


    def init_per_transcript(self, transcript):
        self.transcript_models = [self.models[mp] for mp in transcript.monophones]
        self.states = []
        for model in self.transcript_models:
            self.states += model.states
        self.vectors = transcript.get_vectors()
        self.states_count = len(self.states)
        self.vectors_count = len(self.vectors)
        self.create_state_occupancy_table()


    def calculate_parameters(self):
        for state in self.states:
            state.create_observation_table(self.vectors)
        self.pheno_trans_table = self.calc_transition_table()
        self.forward_table = self.calc_forward_table()
        self.backward_table = self.calc_backward_table()
        self.likelihood = self.calc_likelihood()
        self.time_state_occup_table = self.calc_state_occupancy()


    def apply_changed_values(self):
        new_trans_table = self.calc_new_trans_table()
        self.apply_new_trans_table(new_trans_table)
        self.apply_new_gaussians()


    def create_state_occupancy_table(self):
        self.state_occupancy_table = []
        for state in self.states:
            self.state_occupancy_table.append([lib.NEG_INF] * (len(state.mixtures) + 1))


    def calc_transition_table(self):
        table = []
        for _ in range(self.states_count + 2):
            table.append([lib.NEG_INF] * (self.states_count + 2))

        cursor = 0
        prev_prob = 0
        for model in self.transcript_models:
            prev_prob = self.calc_trans_table_per_model(table, model, cursor, prev_prob)
            cursor += len(model.states)
        return table


    def calc_trans_table_per_model(self, table, model, cursor, prev_prob):
        states_count = len(model.states)
        trans_table = model.transition_table.probabilities

        for j in range(1, states_count + 2):
            table[cursor][cursor + j] = prev_prob + trans_table[0][j]

        for i in range(1, states_count + 2):
            for j in range(1, states_count + 2):
                table[cursor + i][cursor + j] = trans_table[i][j]

        return trans_table[-2][-1]


    def calc_forward_table(self):
        fw_table = []
        for _ in range(self.vectors_count):
            fw_table.append([lib.NEG_INF] * self.states_count)

        for sidx, state in enumerate(self.states):
            fw_prob = self.pheno_trans_table[0][sidx + 1] + state.observ_probs[0]
            fw_table[0][sidx] = fw_prob

        for time in range(1, self.vectors_count):
            for nsidx in range(self.states_count):
                self.calc_forward_per_state(fw_table, time, nsidx)

        return fw_table


    def calc_forward_per_state(self, table, time, nsidx):
        nstate = self.states[nsidx]

        values = []
        for psidx in range(self.states_count):
            fw_prob = table[time - 1][psidx]
            trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
            values.append(fw_prob + trans_prob)
        observ_prob = nstate.observ_probs[time]
        table[time][nsidx] = lib.sum_logs(values) + observ_prob


    def calc_backward_table(self):
        bw_table = []
        for _ in range(self.vectors_count):
            bw_table.append([lib.NEG_INF] * self.states_count)

        for sidx in range(self.states_count):
            bw_table[-1][sidx] = self.pheno_trans_table[sidx + 1][-1]

        for time in range(self.vectors_count - 1, 0, -1):
            for psidx in range(self.states_count):
                self.calc_backward_per_state(bw_table, time, psidx)

        return bw_table


    def calc_backward_per_state(self, table, time, psidx):
        values = []
        for nsidx, nstate in enumerate(self.states):
            trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
            observ_prob = nstate.observ_probs[time]
            bw_prob = table[time][nsidx]
            values.append(trans_prob + bw_prob + observ_prob)
        table[time - 1][psidx] = lib.sum_logs(values)


    def calc_likelihood(self):
        values = []
        for idx in range(self.states_count):
            values.append(self.forward_table[-1][idx] + self.pheno_trans_table[idx + 1][-1])
        return lib.sum_logs(values)


    def calc_state_occupancy(self):
        stocc_table = []
        for state in self.states:
            state_row = []
            for _ in range(len(state.mixtures) + 1):
                state_row.append([lib.NEG_INF] * self.vectors_count)
            stocc_table.append(state_row)

        for time in range(self.vectors_count):
            for sidx in range(self.states_count):
                self.calc_state_occupancy_per_state(stocc_table, time, sidx)

        self.accumulate_state_occupancy(stocc_table)
        return stocc_table


    def calc_state_occupancy_per_state(self, table, time, sidx):
        state = self.states[sidx]

        forward_prob = self.forward_table[time][sidx]
        backward_prob = self.backward_table[time][sidx]
        init_occupancy = forward_prob + backward_prob - self.likelihood
        table[sidx][0][time] = init_occupancy

        observ_prob = state.observ_probs[time]
        for midx, mixture in enumerate(state.mixtures):
            weight = mixture.weight
            part_observ_prob = mixture.observ_probs[time]
            state_occ = init_occupancy + weight + part_observ_prob - observ_prob
            table[sidx][midx + 1][time] = state_occ


    def accumulate_state_occupancy(self, stocc_table):
        for sidx in range(self.states_count):
            state = self.states[sidx]
            for midx in range(len(state.mixtures) + 1):
                state_occupancy = self.state_occupancy_table[sidx][midx]
                prob_sum = lib.sum_logs(stocc_table[sidx][midx] + [state_occupancy])
                self.state_occupancy_table[sidx][midx] = prob_sum


    def calc_new_trans_table(self):
        new_trans_table = []
        for _ in range(self.states_count + 2):
            new_trans_table.append([lib.NEG_INF] * (self.states_count + 2))

        for sidx in range(self.states_count):
            trans_prob = new_trans_table[0][sidx + 1]
            state_occup_prob = self.time_state_occup_table[sidx][0][0]
            new_trans_table[0][sidx + 1] = lib.sum_logs([trans_prob, state_occup_prob])

        for time in range(1, self.vectors_count - 1):
            for psidx in range(self.states_count):
                self.calc_new_trans_per_state(new_trans_table, time, psidx)

        return new_trans_table


    def calc_new_trans_per_state(self, table, time, psidx):
        constant = self.forward_table[time][psidx] - self.likelihood
        for nsidx, nstate in enumerate(self.states):
            trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
            observ_prob = nstate.observ_probs[time + 1]
            bw_prob = self.backward_table[time + 1][nsidx]
            arc_occup = constant + trans_prob + observ_prob + bw_prob
            old_prob = table[psidx + 1][nsidx + 1]
            table[psidx + 1][nsidx + 1] = lib.sum_logs([old_prob, arc_occup])


    def apply_new_trans_table(self, new_table):
        cursor = 0
        exit_prob = 0
        for model in self.transcript_models:
            states_count = len(model.states)

            for nsidx in range(1, states_count + 2):
                prob = self.pheno_trans_table[cursor][cursor + nsidx] - exit_prob

            for psidx in range(1, states_count + 1):
                for nsidx in range(1, states_count + 2):
                    prob = self.pheno_trans_table[cursor + psidx][cursor + nsidx]
                    model.new_table[psidx][nsidx] = prob
            last_state_probs = model.new_table[-2][1:-2]
            exit_prob = lib.get_remaining_prob(last_state_probs)


    def apply_new_gaussians(self):
        for time in range(self.vectors_count):
            for sidx, state in enumerate(self.states):
                self.apply_new_gaussians_per_state(state.mixtures, sidx, time)

    
    def apply_new_gaussians_per_state(self, mixtures, sidx, time):
        vector = self.vectors[time]
        for midx, mixture in enumerate(mixtures):
            state_occup = math.exp(self.time_state_occup_table[sidx][midx + 1][time])
            for gidx, gaussian in enumerate(mixture.gaussians):
                gaussian.new_mean += state_occup * vector.values[gidx]
                gaussian.new_variance += state_occup * (vector.values[gidx] ** 2)
