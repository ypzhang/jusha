#!/usr/bin/python 
#import numpy as np 
#a = np.linspace(0.,10.,100) 
#b = np.sqrt(a) 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 

import csv

def import_text(filename, separator):
    for line in csv.reader(open(filename), delimiter=separator, 
                           skipinitialspace=True):
        if line:
            yield line


def column(matrix, i):
    return [row[i] for row in matrix]


#read data
table = []
for data in import_text('cuMemcpy_direct.dat', ' '):
    table.append(data)


print column(table, 0)[1:]
size = column(table, 1)[1:]

# data
char = column(table, 2)[1:]
float = column(table, 3)[1:]
print float
# start drawing
fig = plt.figure() 
ax = fig.add_subplot(111) 
ax.set_title("cuMemcpy Versus d2d_direct_kernel");
ax.set_xlabel(table[0][0])
ax.set_ylabel('Bandwidth')

fig.add_subplot(ax) 
ax.plot(size, char, linestyle = ':', color = 'k', linewidth = 3) 
ax.plot(size, float, linestyle = ':', color = 'k', linewidth = 3) 
plt.show()
fig.savefig('cudaMemcpy_d2d.eps') 
