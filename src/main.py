import math

import lib

import parsers.hmm_model as hmm_model
import parsers.transcript as transcript
import parsers.vectors as vectors

def accumulate(model, transcripts):
    a = 0
    mean = 0
    variance = 0
    state_occupancy = 0

    transcript = transcripts[0]
    models = [model[mp] for mp in transcript.monophones]
    states = []
    for model in models:
        states += model.states

    pheno_trans_table = get_transcript_transition_table(models, len(states))
    vectors = transcript.get_vectors()
    forward_table = get_forward_table(states, vectors, pheno_trans_table)
    backward_table = get_backward_table(states, vectors, pheno_trans_table)


def get_transcript_transition_table(models, total_states_count):
    table = []
    for _ in range(total_states_count + 2):
        table.append([lib.NEG_INF] * (total_states_count + 2))

    cursor = 0
    table[0][1] = 0
    for model in models:
        states_count = len(model.states)
        trans_table = model.transition_table.probabilities

        for j in range(1, states_count + 2):
            previous_prob = table[cursor][cursor + 1]
            table[cursor][cursor + j] = previous_prob + trans_table[0][j]

        for i in range(1, states_count + 2):
            for j in range(1, states_count + 2):
                table[cursor + i][cursor + j] = trans_table[i][j]

        cursor += states_count

    table[total_states_count][total_states_count + 1] = 0

    return table


def get_forward_table(states, vectors, trans_table):
    fw_table = []
    for _ in range(len(vectors)):
        fw_table.append([lib.NEG_INF] * len(states))

    for sidx, state in enumerate(states):
        observ_prob = state.get_observ_prob(vectors[0])
        fw_prob = trans_table[0][sidx] + observ_prob
        fw_table[0][sidx] = fw_prob

    for time in range(1, len(vectors)):
        vector = vectors[time]
        for sidx, state in enumerate(states):
            prev_vals = [fw_table[time - 1][i] + trans_table[i + 1][sidx + 1] for i in range(len(states))]
            observ_prob = state.get_observ_prob(vector)
            fw_table[time][sidx] = lib.sum_logs(prev_vals) + observ_prob

    return fw_table


def get_backward_table(states, vectors, trans_table):
    bw_table = []
    for _ in range(len(vectors)):
        bw_table.append([lib.NEG_INF] * len(states))

    for sidx in range(len(states)):
        bw_table[-1][sidx] = trans_table[sidx + 1][len(states) + 1]

    for time in range(len(vectors) - 2, 0, -1):
        vector = vectors[time]
        for psidx in range(len(states)):
            values = []
            for nsidx, nstate in enumerate(states):
                trans_prob = trans_table[psidx + 1][nsidx + 1]
                observ_prob = nstate.get_observ_prob(vector)
                bw_prob = bw_table[time + 1][nsidx]
                values.append(trans_prob + bw_prob + observ_prob)

            res = lib.sum_logs(values)
            bw_table[time - 1][psidx] = res
            print(time, psidx, res)

    return bw_table


def get_observation_prob(state, vector):
    return lib.sum_logs([mix.get_observ_prob(vector) for mix in state.mixtures])


def get_indiv_observation_prob(mixture, vector):
    constant = 0.5 * mixture.dimension * math.log(2 * math.pi)
    variances_sum = sum(math.log(gaussian.variance) for gaussian in mixture.gaussians)

    differences_sum = 0
    for i in range(mixture.dimension):
        gaussian = mixture.gaussians[i]
        differences_sum = ((vector.values[i] - gaussian.mean) / gaussian.variance) ** 2
        
    return mixture.weight - constant - variances_sum - 0.5 * differences_sum


def main():
    model = hmm_model.parse()
    transcripts = transcript.parse()

    accumulate(model, transcripts)
    

if __name__ == '__main__':
    main()
