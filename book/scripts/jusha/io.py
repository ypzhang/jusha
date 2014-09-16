import csv

def import_text(filename, separator):
    for line in csv.reader(open(filename), delimiter=separator, 
                           skipinitialspace=True):
        if line:
            yield line


def column(matrix, i):
    return [row[i] for row in matrix]

