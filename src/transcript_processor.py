import lib

class TranscriptProcessor:

    def __init__(self, model):
        self.model = model


    def process(self, transcript):
        self.transcript_models = [self.model[mp] for mp in transcript.monophones]
        self.states = []
        for model in self.transcript_models:
            self.states += model.states
        self.vectors = transcript.get_vectors()
        self.states_count = len(self.states)
        self.vectors_count = len(self.vectors)

        for state in self.states:
            state.create_observation_table(self.vectors)
        
        self.pheno_trans_table = self.calc_transition_table()
        self.forward_table = self.calc_forward_table()
        self.backward_table = self.calc_backward_table()
        self.likelihood = self.calc_likelihood()
        self.time_state_occup_table = self.calc_time_state_occupancy()
        print('PARAMETERS CALCULATION COMPLETE')

        new_trans_table = self.calc_new_trans_table()


    def calc_transition_table(self):
        table = []
        for _ in range(self.states_count + 2):
            table.append([lib.NEG_INF] * (self.states_count + 2))

        cursor = 0
        table[0][1] = 0
        for model in self.transcript_models:
            states_count = len(model.states)
            trans_table = model.transition_table.probabilities

            for j in range(1, states_count + 2):
                previous_prob = table[cursor][cursor + 1]
                table[cursor][cursor + j] = previous_prob + trans_table[0][j]

            for i in range(1, states_count + 2):
                for j in range(1, states_count + 2):
                    table[cursor + i][cursor + j] = trans_table[i][j]

            cursor += states_count

        table[-2][-1] = 0
        return table


    def calc_forward_table(self):
        fw_table = []
        for _ in range(self.vectors_count):
            fw_table.append([lib.NEG_INF] * self.states_count)

        for sidx, state in enumerate(self.states):
            fw_prob = self.pheno_trans_table[0][sidx + 1] + state.observ_probs[0]
            fw_table[0][sidx] = fw_prob

        for time in range(1, self.vectors_count):
            for nsidx, state in enumerate(self.states):
                values = []
                for psidx in range(self.states_count):
                    fw_prob = fw_table[time - 1][psidx]
                    trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
                    values.append(fw_prob + trans_prob)
                observ_prob = state.observ_probs[time]
                fw_table[time][nsidx] = lib.sum_logs(values) + observ_prob

        return fw_table


    def calc_backward_table(self):
        bw_table = []
        for _ in range(self.vectors_count):
            bw_table.append([lib.NEG_INF] * self.states_count)

        for sidx in range(self.states_count):
            bw_table[-1][sidx] = self.pheno_trans_table[sidx + 1][-1]

        for time in range(self.vectors_count - 1, 0, -1):
            for psidx in range(self.states_count):
                values = []
                for nsidx, nstate in enumerate(self.states):
                    trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
                    observ_prob = nstate.observ_probs[time]
                    bw_prob = bw_table[time][nsidx]
                    values.append(trans_prob + bw_prob + observ_prob)
                bw_table[time - 1][psidx] = lib.sum_logs(values)

        return bw_table


    def calc_likelihood(self):
        values = []
        for idx in range(self.states_count):
            values.append(self.forward_table[-1][idx] + self.pheno_trans_table[idx + 1][-1])
        return lib.sum_logs(values)


    def calc_time_state_occupancy(self):
        stocc_table = []
        for state in self.states:
            state_row = []
            for _ in range(len(state.mixtures) + 1):
                state_row.append([lib.NEG_INF] * self.vectors_count)
            stocc_table.append(state_row)

        for time in range(self.vectors_count):
            for sidx, state in enumerate(self.states):
                forward_prob = self.forward_table[time][sidx]
                backward_prob = self.backward_table[time][sidx]
                init_stocc = forward_prob + backward_prob - self.likelihood
                stocc_table[sidx][0][time] = (init_stocc)

                observ_prob = state.observ_probs[time]
                for midx, mixture in enumerate(state.mixtures):
                    weight = mixture.weight
                    part_observ_prob = mixture.observ_probs[time]
                    state_occ = init_stocc + weight + part_observ_prob - observ_prob
                    stocc_table[sidx][midx + 1][time] = state_occ

        return stocc_table


    def calc_new_trans_table(self):
        new_trans_table = []
        for _ in range(self.states_count + 2):
            new_trans_table.append([lib.NEG_INF] * (self.states_count + 2))

        for sidx in range(self.states_count):
            new_trans_table[0][sidx + 1] = self.time_state_occup_table[sidx][0][0]

        for time in range(1, self.vectors_count - 1):
            for psidx in range(self.states_count):
                fw_prob = self.forward_table[time][psidx]
                for nsidx, nstate in enumerate(self.states):
                    trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
                    observ_prob = nstate.observ_probs[time + 1]
                    bw_prob = self.backward_table[time + 1][nsidx]
                    arc_occup = fw_prob + trans_prob + observ_prob + bw_prob - self.likelihood
                    old_prob = new_trans_table[psidx + 1][nsidx + 1]
                    new_trans_table[psidx + 1][nsidx + 1] = lib.sum_logs([old_prob, arc_occup])

        return new_trans_table