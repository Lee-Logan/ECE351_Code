# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 20:20:02 2019

@author: Logan Lee
ECE 351-52
"""

import numpy as np
import time
import scipy.signal as sig
import math as m
import matplotlib.pyplot as plt


t=1
print(t)
print("t =",t)
print('t =',t,"seconds")
print('t is now =',t/3,'\n. . .and can be rounded using `round()`',round(t/3,4))
print(3**2)
# This is a comment, and the following statement will not be executed:
# print(t+5)

import numpy
list1 =[0,1,2,3]
print('list1:',list1)
list2 = [[0],[1],[2],[3]]
print('list2:', list2)
list3 = [[0,1],[2,3]]
print('list3:',list3)
array1 = numpy.array([0,1,2,3])
print('array1:',array1)
array2 = numpy.array([[0],[1],[2],[3]])
print('array2:',array2)
array3 = numpy.array([[0,1],[2,3]])
print('array3:',array3)

print(numpy.pi)
print(np.pi)

from numpy import pi
print(pi)

print(np.arange(4),'\n',
      np.arange(0,2,0.5),'\n',
      np.linspace(0,1.5,4))
list1 = [1,2,3,4,5]
array1 = np.array(list1)
print('list1:',list1[0],list1[4])
print('array1:',array1[0],array1[4])

array2 = np.array([[1,2,3,4,5],[6,7,8,9,10]])
list2 = list(array2)
print('array2:',array2[0,2],array2[1,4])
print('list2:',list2[0],list2[1])
print(array2[:,2],
array2[0,:])
print('1x3:',np.zeros(3))
print('2x2:',np.zeros((2,2)))
print('2x3:',np.ones((2,3)))
steps = 0.1
x = np.arange(-2,2+steps,steps)
y1 = x+2
y2 = x**2

plt.figure(figsize=(12,8))
plt.subplot(3,1,1)
plt.plot(x,y1)
plt.title('Sample Plots for Lab 1')
plt.grid(True)
plt.subplot(3,1,2)
plt.plot(x,y2)
plt.ylabel('subplot 2')
plt.grid(which='both')
plt.subplot(3,1,3)
plt.plot(x,y1,'--r',label='y1')
plt.plot(x,y2,'o',label='y2')
plt.axis([-2.5,2.5,-0.5,4.5])
plt.grid(True)
plt.legend(loc='lower right')
plt.xlabel('x')
plt.ylabel('Subplot 3')

plt.show()

cRect=2+3j
print(cRect)
cPol=abs(cRect)*np.exp(1j*np.angle(cRect))
print(cPol)
cRect2=np.real;(cPol) + 1j*np.imag(cPol)
print(cRect2)
print(numpy.sqrt(3*5 - 5*5))

print(numpy.sqrt(3*5 - 5*5 + 0j))
