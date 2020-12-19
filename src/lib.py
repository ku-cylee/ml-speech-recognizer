import math

def sum_logs(values):
    min_value = min(values)
    return min_value + math.log(sum(math.exp(value - min_value) for value in values))
