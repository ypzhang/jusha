#!/usr/bin/python
import numpy as np 
a = np.linspace(0.,10.,100) 
b = np.sqrt(a) 
c = a
import matplotlib 
matplotlib.use('Agg') 
import matplotlib.pyplot as plt 
fig = plt.figure() 
ax = fig.add_subplot(111) 

ax.set_title("");
ax.set_xlabel('x label')
ax.set_ylabel('y label')
fig.add_subplot(ax) 
ax.plot(a, b, linestyle = ':', color = 'k', linewidth = 3) 
ax.plot(a, c, linestyle = '-', color = 'k', linewidth = 3) 
plt.show()
fig.savefig('broken.eps') 
