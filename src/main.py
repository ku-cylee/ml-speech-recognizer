import math

import lib

import parsers.hmm_model as hmm_model
import parsers.transcript as transcript
import parsers.vectors as vectors

class Accumulator:

    def __init__(self, model):
        self.model = model


    def process_transcript(self, transcript):
        self.transcript_models = [self.model[mp] for mp in transcript.monophones]
        self.states = []
        for model in self.transcript_models:
            self.states += model.states
        self.vectors = transcript.get_vectors()
        self.states_count = len(self.states)
        self.vectors_count = len(self.vectors)
        
        self.pheno_trans_table = self.calc_transition_table()
        print('TRANS TABLE COMPLETE')
        self.forward_table = self.calc_forward_table()
        print('FORWARD TABLE COMPLETE')
        self.backward_table = self.calc_backward_table()
        print('BACKWARD TABLE COMPLETE')
        self.likelihood = self.calc_likelihood()


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
            observ_prob = state.get_observ_prob(self.vectors[0])
            fw_prob = self.pheno_trans_table[0][sidx + 1] + observ_prob
            fw_table[0][sidx] = fw_prob

        for time in range(1, self.vectors_count):
            vector = self.vectors[time]
            for nsidx, state in enumerate(self.states):
                values = []
                for psidx in range(self.states_count):
                    fw_prob = fw_table[time - 1][psidx]
                    trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
                    values.append(fw_prob + trans_prob)
                observ_prob = state.get_observ_prob(vector)
                fw_table[time][nsidx] = lib.sum_logs(values) + observ_prob

        return fw_table


    def calc_backward_table(self):
        bw_table = []
        for _ in range(self.vectors_count):
            bw_table.append([lib.NEG_INF] * self.states_count)

        for sidx in range(self.states_count):
            bw_table[-1][sidx] = self.pheno_trans_table[sidx + 1][-1]

        for time in range(self.vectors_count - 2, 0, -1):
            vector = self.vectors[time]
            for psidx in range(self.states_count):
                values = []
                for nsidx, nstate in enumerate(self.states):
                    trans_prob = self.pheno_trans_table[psidx + 1][nsidx + 1]
                    observ_prob = nstate.get_observ_prob(vector)
                    bw_prob = bw_table[time + 1][nsidx]
                    values.append(trans_prob + bw_prob + observ_prob)
                bw_table[time - 1][psidx] = lib.sum_logs(values)

        return bw_table


    def calc_likelihood(self):
        values = []
        for idx in range(self.states_count):
            values.append(self.forward_table[-1][idx] + self.pheno_trans_table[idx + 1][-1])
        return lib.sum_logs(values)


def main():
    model = hmm_model.parse()
    transcripts = transcript.parse()

    accumulator = Accumulator(model)
    ts = transcripts[0]
    print('Accumulating Transcript: {0}'.format(ts.filename))
    accumulator.process_transcript(ts)


if __name__ == '__main__':
    main()
