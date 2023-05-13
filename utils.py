def count_min_max(data, type):
    values = []
    for year in data:
        value = data[year]
        values.append(value)

    if type == 'min':
        return min(values)
    else:
        return max(values)


def count_min_max_range(data, type):
    values = []
    for year in data:
        value = data[year]
        values.append(value)

    if type == 'min':
        return min(values)
    else:
        return max(values)
