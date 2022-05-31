# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 20:37:09 2022

@author: Sasuke
"""

import scipy.signal as sp
import numpy as np
import matplotlib.pyplot as plt

#Helper functions

def gib_Fs_Xs_TF(freq = 1.5, decay = 0.5, Fs = None, Xs = False, TF = False):
    if not Xs and not TF:
        return sp.lti([1,decay], [1,2*decay, freq**2 + decay**2])
    elif TF:
        return sp.lti([1], [1, 0, freq**2])
    
    else:
        assert Fs != None, "Enter Fs"
        p = np.polymul(np.poly1d([1,0,freq**2]),
                       np.poly1d([1,2*decay, freq**2 + decay**2]))
        return sp.lti([1,decay], p)

def ft(freq ,decay,t):
    return np.cos(freq*t)*np.exp(-decay*t)

def inSignal(t):
    return np.cos(10**3*t) - np.cos(10**6*t)

def TF_5_plotting(C = 1e-6, R = 100, L = 1e-6, bode = False, inSignal = None):
    H = sp.lti([1], [L*C, R*C, 1])
    if bode:
        w,S, phi = H.bode()
        plt.figure(figsize = (70,70))
        plt.subplot(10,8,1)
        plt.title("Magnitude plot")
        plt.ylabel(r"$|H(j\omega)|\rightarrow$")
        plt.xlabel(r"$\omega\rightarrow$")
        plt.semilogx(w, S, label = "Magnitude")
        plt.grid();plt.legend()
        plt.subplot(10,8,2)
        plt.title("Phase plot")
        plt.ylabel(r"\phi\rightarrow")
        plt.xlabel(r"\omega\rightarrow")
        plt.semilogx(w,phi, label ="Phase")
        plt.grid();plt.legend()
        plt.show()
    else:
        assert inSignal != None, "Provide input signal"
        t1 = np.linspace(0,30e-6,150)
        t1,y1,_ = sp.lsim(H, inSignal(t1), T = t1)
        t2 = np.linspace(0,15e-3, 10000000)
        t2,y2,_ = sp.lsim(H, inSignal(t2), T = t2)
        plt.figure(figsize = (70,70))
        plt.subplot(10,8,1)
        plt.title(r"$V_o$ till $15\mu s$")
        plt.ylabel(r"$V_o(t)\rightarrow$")
        plt.xlabel(r"$t\rightarrow$")
        plt.plot(t1, y1, label = "y(t)")
        plt.grid();plt.legend()
        plt.subplot(10,8,2)
        plt.title(r"$V_o$ till $30ms$")
        plt.ylabel(r"$V_o(t)\rightarrow$")
        plt.xlabel(r"$t\rightarrow$")
        plt.plot(t2,y2, label = "y(t)")
        plt.grid();plt.legend()
        plt.show()
        
### Question 1


Fs =gib_Fs_Xs_TF()
Xs = gib_Fs_Xs_TF(decay = 0.5, Fs = Fs,Xs = True)
t = np.linspace(1,50, 300) #numpy defaults to 50 points
t, xt = sp.impulse(Xs, T = t)
#plotting

plt.figure(figsize = (10,8))
plt.title("x(t) vs t")
plt.xlabel(r"$x(t)\rightarrow$")
plt.ylabel(r"$t\rightarrow$")
plt.plot(t, xt, label = "$\gamma = 0.5$")
plt.grid();plt.legend()
plt.show()

### Question 2

Fs = gib_Fs_Xs_TF(decay = 0.05)
Xs = gib_Fs_Xs_TF(decay = 0.05, Fs = Fs, Xs = True)
t = np.linspace(1,50, 300) #numpy defaults to 50 points
t, xt = sp.impulse(Xs, T = t)

#plotting

plt.figure(figsize = (10,8))
plt.title("x(t) vs t")
plt.xlabel(r"$x(t)\rightarrow$")
plt.ylabel(r"$t\rightarrow$")
plt.plot(t, xt, label = "$\gamma = 0.05$")
plt.grid();plt.legend()
plt.show()

### Question 3

for freq in np.arange(1.4, 1.65, 0.05):
    t = np.linspace(1,50, 300)
    TF = gib_Fs_Xs_TF(freq = 1.5, decay = 0.05, TF = True)
    t,y,_ = sp.lsim(TF, U = ft(freq, 0.05, t), T = t)
    plt.figure(figsize = (10,8))
    plt.title("x(t) vs t")
    plt.xlabel(r"$x(t)\rightarrow$")
    plt.ylabel(r"$t\rightarrow$")
    plt.plot(t, y, label = f"Frequency = {freq}")
    plt.grid();plt.legend()
    plt.show()
plt.figure(figsize = (10,8))
plt.title("x(t) vs t")
plt.xlabel(r"$x(t)\rightarrow$")
plt.ylabel(r"$t\rightarrow$")
for freq in np.arange(1.4, 1.65, 0.05):
    t = np.linspace(1,50, 300)
    TF = gib_Fs_Xs_TF(freq = freq, decay = 0.05, TF = True)
    t,y,_ = sp.lsim(TF, U = ft(freq, 0.05, t), T = t)
    plt.plot(t, y, label = f"Frequency = {freq}")
plt.grid();plt.legend()
plt.show()

### Question 4

Xs = sp.lti([1, 0, 2], [1, 0, 3, 0])
Ys = sp.lti([2], [1, 0, 3, 0])
#For x(t)
t1 = np.linspace(0, 20, 100)
t1,xt = sp.impulse(Xs, T = t1)
#For y(t)
t2 = np.linspace(0, 20, 100)
t2, yt = sp.impulse(Ys, T = t2)


plt.figure(figsize = (10,8))
plt.title("x(t) & y(t) vs t")
plt.xlabel(r"$x(t) & y(t)\rightarrow$")
plt.ylabel(r"$t\rightarrow$")
plt.plot(t1,xt, label = "x(t)")
plt.plot(t2 ,yt, label = "y(t)")
plt.grid();plt.legend()
plt.show()

### Question 5

TF_5_plotting(bode = True)

### Quesrtion 6

TF_5_plotting(inSignal = inSignal)

plt.close()