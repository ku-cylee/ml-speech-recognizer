import math

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
    states_count = sum(len(model.states) for model in models)
    model_transition_table = get_model_transition_table(models, states_count)


def get_model_transition_table(models, total_states_count):
    table = []
    for _ in range(total_states_count + 2):
        table.append([0] * (total_states_count + 2))

    cursor = 0
    for model in models:
        states_count = len(model.states)
        trans_table = model.transition_table.probabilities

        for j in range(1, states_count + 2):
            previous_prob = table[cursor][cursor + 1]
            table[cursor][cursor + j] = previous_prob * trans_table[0][j]

        for i in range(1, states_count + 2):
            for j in range(1, states_count + 2):
                table[cursor + i][cursor + j] = trans_table[i][j]

        cursor += states_count

    table[total_states_count][total_states_count + 1] = 1

    return table


def main():
    model = hmm_model.parse()
    transcripts = transcript.parse()

    accumulate(model, transcripts)
    

if __name__ == '__main__':
    main()
