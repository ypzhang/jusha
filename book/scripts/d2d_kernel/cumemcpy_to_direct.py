#!/usr/bin/python 
import numpy as np 
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


def to_num(s):
    try:
        return int(s)
    except ValueError:
        return float(s)

def to_float(s):
    try:
        return float(s)
    except ValueError:
        return int(s)



def column(matrix, i):
    return [row[i] for row in matrix]

def bandwidth(timings, sizes):
    result = []
    for i in range(0, len(timings)):
        result.append((2*to_float(sizes[i]))/(to_float(timings[i])*1000000000.0))
    return result

        
#read data
table = []
for data in import_text('./0_cudamemcpy.dat', ' '):
    table.append(data)


#print column(table, 0)[1:]
size = column(table, 1)[1:]

size_string = column(table, 0)[1:]
#print size_string
# data
char_t = column(table, 2)[1:]
short_t = column(table, 3)[1:]
float_t = column(table, 4)[1:]
double_t = column(table, 5)[1:]
float3_t = column(table, 6)[1:]
float4_t = column(table, 7)[1:]

char_bw = bandwidth(char_t, size)
short_bw = bandwidth(short_t, size)
float_bw = bandwidth(float_t, size)
double_bw = bandwidth(double_t, size)
float3_bw = bandwidth(float3_t, size)
float4_bw = bandwidth(float4_t, size)


# read other table
di_table = []
for di_data in import_text('./1_direct.dat', ' '):
    di_table.append(di_data)


#print column(table, 0)[1:]

#size_string = column(table, 0)[1:]
#print size_string
# data
di_char_t = column(di_table, 2)[1:]
di_short_t = column(di_table, 3)[1:]
di_float_t = column(di_table, 4)[1:]
di_double_t = column(di_table, 5)[1:]
di_float3_t = column(di_table, 6)[1:]
di_float4_t = column(di_table, 7)[1:]

di_char_bw   = bandwidth(di_char_t, size)
di_short_bw  = bandwidth(di_short_t, size)
di_float_bw  = bandwidth(di_float_t, size)
di_double_bw = bandwidth(di_double_t, size)
di_float3_bw = bandwidth(di_float3_t, size)
di_float4_bw = bandwidth(di_float4_t, size)


size_np = np.array(size)
# normalize the size
for i in range(0, len(size)):
    size_np[i] = i+1
#    size_np[len(size)-1-i] = to_num(to_num(size_np[len(size)-1-i])/to_num(size_np[0])) #to_float(size[i])/to_float(size[0])


#print to_float(size[11])
#print to_float(float4_t[11])
#print (to_float(2*sizes[i])/(to_float(timings[i])*1000000000.0))

#print char_bw
#print float_bw
#print float
# start drawing
fig = plt.figure() 
ax = fig.add_subplot(111) 
ax.set_title("cuMemcpy v.s. d2d_direct_kernel");
ax.set_xlabel(table[0][0])
ax.set_ylabel('Bandwidth (GB/sec)')

#print len(size_string)
#print len(char_bw)
fig.add_subplot(ax) 
#ax.set_ylim([180,260])
print size_np
print size_string

#ax.set_xticklabels(size_np, range(len(size_np)))
ax.set_xticklabels(size_string)
#fig.xticks(size_np, size_string)
#ax.set_xticks(size_np, ('128K', '256K', '512K', '1M', '2M', '4M', '8M', '16M', '32M', '64M'))

#ax.set_autoscaley_on(False)
ax.plot(size_np, char_bw, linestyle = '-', color = 'blue', marker='o', linewidth = 1, label='cudaMemcpy') 
#ax.plot(size, short_bw, linestyle = '-', color = 'red', linewidth = 1, label='cudaMemcpy_short') 
#ax.plot(size, float_bw, linestyle = '-', color = 'c', linewidth = 1, label='cudaMemcpy_float') 
#ax.plot(size, double_bw, linestyle = '-', color = 'm', linewidth = 1, label='cudaMemcpy_double') 
#ax.plot(size, float3_bw, linestyle = '-', color = 'k', linewidth = 1, label='cudaMemcpy_float3') 
#ax.plot(size, float4_bw, linestyle = '-', color = 'y', linewidth = 1, label='cudaMemcpy_float4') 


ax.plot(size_np, di_char_bw, linestyle = ':', color = 'blue', marker='o', linewidth = 2, label='d2d_direct_char') 
ax.plot(size_np, di_short_bw, linestyle = ':', color = 'red', marker='s', linewidth = 2, label='d2d_direct_short') 
ax.plot(size_np, di_float_bw, linestyle = ':', color = 'c', marker='p', linewidth = 2, label='d2d_direct_float') 
ax.plot(size_np, di_double_bw, linestyle = ':', color = 'm', marker='*', linewidth = 2, label='d2d_direct_double') 
ax.plot(size_np, di_float3_bw, linestyle = ':', color = 'k', marker='h', linewidth = 2, label='d2d_direct_float3') 
ax.plot(size_np, di_float4_bw, linestyle = ':', color = 'y', marker='x', linewidth = 2, label='d2d_direct_float4') 

size_num=range(len(size))
#print size_num
print size_string



box = ax.get_position()
ax.set_position([box.x0, box.y0+box.height*0.1, box.width, box.height*0.9])
#ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol = 6, fancybox = True, shadow = True, prop={'size':9}, )
#ax.legend(loc='upper center', ncol = 3, fancybox = True, shadow = True, prop={'size':9}, )
#ax.legend(loc='upper left', ncol = 1, fancybox = True, shadow = True, prop={'size':9}, )


ax.legend(loc='upper center', ncol = 4, bbox_to_anchor=(0.5,-0.1),  fancybox = True, shadow = True, prop={'size':9}, )


plt.show()
fig.savefig('cudaMemcpy_vs_d2d.pdf') 
