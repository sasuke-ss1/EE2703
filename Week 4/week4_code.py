# -*- coding: utf-8 -*-
"""week4_code.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jGNNwiIcodo0UILECzos5PZjX_kAmr0B

### INITIALIZATIONS
"""

import numpy as np
import scipy as sp
from scipy import integrate
import matplotlib.pyplot as plt
import math

#Exponential function that takes both scalar and vector as input
def exp(x):
    return np.exp(x)

#Nested cos function that takes both scalar and vector as input
def cc(x):
    return np.cos(np.cos(x))


#helper
a_s = np.array([True]+ [True if i%2 != 0 else False for i in range(1,51)])
b_s = np.array([False] + [True if i%2 == 0 else False for i in range(1,51)])

"""### QUESTION 1"""

#Using 100 points in between -2pi and 4pi for plot
x = np.linspace(-2*math.pi, 4*math.pi, 99)
temp = exp(np.linspace(0,2*math.pi,33)).reshape(-1,1)
ffit = np.concatenate((temp,temp,temp), axis = 0)
fig = plt.figure(figsize=(70,70))
plt.subplot(10,10,1)
plt.title("Plot of $e^{x}$ in semilog y scale")
plt.plot(x, exp(x), label = "$e^{x}$")
plt.plot(x, ffit, linestyle = "dashed" ,label = "Fourier fit")
plt.grid()
plt.legend()
plt.ylabel("$e^{x}$", fontsize =15);plt.xlabel("x", fontsize = 15)
plt.yscale("log")
plt.subplot(10,10,2)
plt.title("Plot of $cos(cos(x))$ in linear scale")
plt.plot(x, cc(x), label = "cos(cos(x))")
plt.plot(x, cc(x),linestyle = "dashed" ,label = "Fourier fit")
plt.xlabel("x", fontsize =15);plt.ylabel("$cos(cos(x))$", fontsize =15)
plt.legend()
plt.grid()
plt.show()

"""### QUESTION 2"""

a = np.zeros((26,2))
b_ = np.zeros((26,2))
a[0,:]  = 1/(2*math.pi)*np.array([integrate.quad(exp, 0, 2*math.pi)[0], integrate.quad(cc, 0, 2*math.pi)[0]])
plotting = np.zeros((1,2))
plotting[0,:] = a[0,:]
for i in range(1, 26):
    a[i,:] = 1/math.pi*np.array([integrate.quad(lambda x, k: exp(x)*math.cos(k*x), 0, 2*math.pi, args=(i,))[0]
              , integrate.quad(lambda x, k: cc(x)*math.cos(k*x), 0, 2*math.pi, args=(i,))[0]])
    b_[i,:] = 1/math.pi*np.array([integrate.quad(lambda x, k: exp(x)*math.sin(k*x), 0, 2*math.pi, args=(i,))[0]
              , integrate.quad(lambda x, k: cc(x)*math.sin(k*x), 0, 2*math.pi, args=(i,))[0]])
    plotting = np.concatenate((plotting,a[i,:].reshape(-1,2), b_[i,:].reshape(-1,2)))

"""### QUESTION 3"""

plt.figure(figsize=(70,70))
plt.subplot(10,10,1)
plt.title("Coefficients of fourier series of $e^{x}$ in semilog y scale")
plt.semilogy(np.abs(a[:,0]), "ro", label = "$a_n$")
plt.semilogy(range(1,26), np.abs(b_[1:,0]), "bo", label = "$b_n$")
plt.grid()
plt.legend()
plt.subplot(10,10,2)
plt.loglog(np.abs(a[:,0]), "ro", label = "$a_n$")
plt.loglog(range(1,26), np.abs(b_[1:,0]), "bo", label = "$b_n$")
plt.title("Coefficients of fourier series of $e^{x}$ in loglog scale")
plt.grid()
plt.legend()
plt.show()
plt.figure(figsize=(70,70))
plt.subplot(10,10,1)
plt.semilogy(np.abs(a[:,1]), "ro", label = "$a_n$")
plt.semilogy(range(1,26), np.abs(b_[1:,1]), "bo", label = "$b_n$")
plt.grid()
plt.title("Coefficients of fourier series of $cos(cos(x))$ in semilog y scale")
plt.legend()
plt.subplot(10,10,2)
plt.loglog(np.abs(a[:,1]), "ro", label = "$a_n$")
plt.loglog(range(1,26), np.abs(b_[1:,1]), "bo", label = "$b_n$")
plt.grid()
plt.title("Coefficients of fourier series of $cos(cos(x})$ in loglog scale")
plt.legend()
plt.show()

"""### QUESTION 4"""

x = np.linspace(0,2*math.pi,400, endpoint = False).reshape(-1,1)
b = np.c_[exp(x), cc(x)]

def get_A(dim1, dim2, x):
    A = np.ones((dim1,1))
    tmp = 1
    for i in range(1,dim2):
        if(i%2 != 0):
            A = np.c_[A, np.multiply(np.cos(tmp*x).reshape(-1,1),np.ones((dim1,1)))]
        else:
            A = np.c_[A, np.multiply(np.sin(tmp*x).reshape(-1,1),np.ones((dim1,1)))]
            tmp+=1
    return A
A = get_A(400,51,x)

"""### QUESTION 5"""

c_0 = sp.linalg.lstsq(A,b[:,0])[0].reshape(-1,1)
c_1 = sp.linalg.lstsq(A,b[:,1])[0].reshape(-1,1)

c_0_a = c_0[a_s,:];c_0_b = c_0[b_s,:]

c_1_a = c_1[a_s,:];c_1_b = c_1[b_s,:]
plt.show()
plt.figure(figsize=(70,70))
plt.subplot(10,10,1)
plt.title("Plot of least squre fit vs true fourier coefficients")
plt.semilogy(np.abs(a[:,0]), "ro", label = "True $a_n$")
plt.semilogy(range(1,26), np.abs(b_[1:,0]), "ro", label = "True $b_n$")
plt.semilogy(np.abs(c_0_a[:,0]), "go", label = "Estimated $a_n$")
plt.semilogy(range(1,26), np.abs(c_0_b[:,0]), "go", label = "Estimated $b_n$")
plt.grid()
plt.legend()
plt.subplot(10,10,2)
plt.title("Plot of least squre fit vs true fourier coefficients")
plt.loglog(np.abs(a[:,0]), "ro", label = "True $a_n$")
plt.loglog(range(1,26), np.abs(b_[1:,0]), "ro", label = "True $b_n$")
plt.loglog(np.abs(c_0_a[:,0]), "go", label = "Estimated $a_n$")
plt.loglog(range(1,26), np.abs(c_0_b[:,0]), "go", label = "Estimated $b_n$")
plt.grid()
plt.legend()
plt.show()
plt.figure(figsize=(70,70))
plt.subplot(10,10,1)
plt.title("Plot of least squre fit vs true fourier coefficients")
plt.semilogy(np.abs(a[:,1]), "ro", label = "True $a_n$")
plt.semilogy(range(1,26), np.abs(b_[1:,1]), "ro", label = "True $b_n$")
plt.semilogy(np.abs(c_1_a[:,0]), "go", label = "Estimated $a_n$")
plt.semilogy(range(1,26), np.abs(c_1_b[:,0]), "go", label = "Estimated $b_n$")
plt.grid()
plt.legend()
plt.subplot(10,10,2)
plt.title("Plot of least squre fit vs true fourier coefficients")
plt.loglog(np.abs(a[:,1]), "ro", label = "True $a_n$")
plt.loglog(range(1,26), np.abs(b_[1:,1]), "ro", label = "True $b_n$")
plt.loglog(np.abs(c_1_a[:,0]), "go", label = "Estimated $a_n$")
plt.loglog(range(1,26), np.abs(c_1_b[:,0]), "go", label = "Estimated $b_n$")
plt.grid()
plt.legend()
plt.show()

"""### QUESTION 6"""

err = abs(plotting - np.c_[c_0,c_1])
plt.plot(err[:,0],"bo", label = "Error")
plt.title("Error in estimated vs true value")
plt.legend()
plt.show()
print(f"The max error is {err.max(axis = 0)[0]}")
plt.plot(err[:,1], "bo", label = "Error")
plt.title("Error in estimated vs true value")
plt.legend()
plt.show()
print(f"The max error is {err.max(axis = 0)[1]}")

"""### QUESTION 7"""

f_1 = np.dot(A,c_0)
f_2 = np.dot(A,c_1)
fig = plt.figure(figsize=(70,70))
plt.subplot(10,10,1)
plt.title("Estimated vs Actual graph")
plt.plot(x, exp(x), "k", label = "True")
plt.plot(x, f_1, "go", alpha = 0.3, label = "Estimated")
plt.legend()
plt.grid()
plt.ylabel("$e^{x}$", fontsize =15);plt.xlabel("x", fontsize = 15)
plt.yscale("log")
plt.subplot(10,10,2)
plt.title("Estimated vs Actual graph")
plt.plot(x, cc(x),"k", label = "True")
plt.plot(x,f_2, "go", alpha = 0.3, label = "Estimated")
plt.xlabel("x", fontsize =15);plt.ylabel("$cos(cos(x))$", fontsize =15)
plt.legend()
plt.grid()
plt.show()