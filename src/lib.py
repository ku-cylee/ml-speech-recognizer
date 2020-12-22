import math

NEG_INF = -1 * math.inf

def refined_log(value):
    if value == 0:
        return NEG_INF
    else:
        return math.log(value)

def sum_logs(values):
    filtered_values = [val for val in values if val != NEG_INF]
    if not filtered_values:
        return NEG_INF

    max_value = max(filtered_values)
    return max_value + math.log(sum(math.exp(value - max_value) for value in filtered_values))


def get_remaining_prob(probs):
    probs_sum = sum_logs(probs)
    return refined_log(1 - math.exp(probs_sum))
