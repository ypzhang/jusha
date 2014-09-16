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


def num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)


def column(matrix, i):
    return [row[i] for row in matrix]

def bandwidth(timings, sizes):
    result = []
    for i in range(0, len(timings)):
        result.append(num(2*sizes[i])/(num(timings[i])*1000000000.0))
    return result
#    print result

#        print num(sizes[i])
#        print timings[i]
#       result.append(float(sizes[i])/float(timings[i]))
#        return result

        
#read data
table = []
for data in import_text('cuMemcpy_direct.dat', ' '):
    table.append(data)


#print column(table, 0)[1:]
size = column(table, 1)[1:]

# data
char = column(table, 2)[1:]
float = column(table, 3)[1:]
char_bw = bandwidth(char, size)
float_bw = bandwidth(float, size)

print size
print char_bw
print float_bw
#print float
# start drawing
fig = plt.figure() 
ax = fig.add_subplot(111) 
ax.set_title("cuMemcpy Versus d2d_direct_kernel");
ax.set_xlabel(table[0][0])
ax.set_ylabel('Bandwidth')

fig.add_subplot(ax) 
ax.plot(size, char_bw, linestyle = ':', color = 'k', linewidth = 3) 
ax.plot(size, float_bw, linestyle = ':', color = 'k', linewidth = 3) 
plt.show()
fig.savefig('cudaMemcpy_d2d.eps') 
