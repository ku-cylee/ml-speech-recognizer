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

    model_transition_table = get_model_transition_table(models, len(states))
    vectors = transcript.get_vectors()
    forward_table = get_forward_table(states, vectors, model_transition_table)


def get_model_transition_table(models, total_states_count):
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
    fw_table.append([])

    for sidx, state in enumerate(states):
        observ_prob = get_observation_prob(state, vectors[0])
        fw_prob = trans_table[0][sidx] + observ_prob
        fw_table[0].append(fw_prob)

    for time in range(1, len(vectors)):
        vector = vectors[time]
        fw_table.append([])
        for sidx, state in enumerate(states):
            prev_vals = [fw_table[time - 1][i] + trans_table[i][sidx] for i in range(len(states))]
            observ_prob = get_observation_prob(state, vector)
            fw_table[time].append(lib.sum_logs(prev_vals) + observ_prob)

    return fw_table


def get_observation_prob(state, vector):
    return lib.sum_logs([get_indiv_observation_prob(mixture, vector) for mixture in state.mixtures])


def get_indiv_observation_prob(mixture, vector):
    log_weight = math.log(mixture.weight)
    constant = 0.5 * mixture.dimension * math.log(2 * math.pi)
    variances_sum = sum(math.log(gaussian.variance) for gaussian in mixture.gaussians)

    differences_sum = 0
    for i in range(mixture.dimension):
        gaussian = mixture.gaussians[i]
        differences_sum = ((vector.values[i] - gaussian.mean) / gaussian.variance) ** 2
        
    return log_weight - constant - variances_sum - 0.5 * differences_sum


def main():
    model = hmm_model.parse()
    transcripts = transcript.parse()

    accumulate(model, transcripts)
    

if __name__ == '__main__':
    main()
