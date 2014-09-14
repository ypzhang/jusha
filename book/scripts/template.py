#!/usr/bin/env python 
import numpy as np 
a = np.linspace(0.,10.,100) 
b = np.sqrt(a) 
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
fig = plt.figure() 
ax = fig.add_subplot(111) 
fig.add_subplot(ax) 
ax.plot(a, b, linestyle = ':', color = 'k', linewidth = 3) 
fig.savefig('broken.pdf') 
