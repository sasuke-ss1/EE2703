# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 12:03:48 2022

@author: Sasuke
"""

import sympy as simp
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

s = simp.symbols("s")

def coeff(expr, var = s):
    num, denom = expr.as_numer_denom()
    num = [float(i) for i in simp.Poly(num, var).all_coeffs()]
    denom = [float(i) for i in simp.Poly(denom, var).all_coeffs()]
    return num, denom

def inSignal(t):
    return (np.sin(2000*np.pi*t) + np.cos(2e6*np.pi*t))*(t>0)

def lowpass(R1,R2,C1,C2,G,Vi):
    
    A = simp.Matrix([
        [0,0,1,-1/G],
        [-1/(1+s*R2*C2), 1, 0, 0],
        [0, -G, G, 1],
        [-1/R1 - 1/R2 -s*C1, 1/R2, 0, s*C1]
        ])
    b = simp.Matrix([0,0,0,-Vi/R1])
    V = A.inv()*b
    
    return A, V, b

def highpass(R1,R3,C1,C2,G,Vi):
    
    A=simp.Matrix([[0,-1,0,1/G],
        [s*C2*R3/(s*C2*R3+1),0,-1,0],
        [0,G,-G,1],
        [-s*C2-1/R1-s*C1,0,s*C2,1/R1]])
    b=simp.Matrix([0,0,0,-Vi*s*C1])
    
    V = A.inv()*b
    
    return (A,V,b)

def indampedS(t,decay=5e4,freq=1e8):
    return np.cos(2*np.pi*freq*t)*np.exp(-decay*t) * (t>0)


R1, R2, C1, C2, G, Vi = 1e4, 1e4, 1e-9, 1e-9, 1.586, 1
A,V,b=lowpass(R1, R2, C1, C2, G, Vi)
print(f"G = {G}")
Vo = V[3]
print(Vo)
ww = np.logspace(0, 8, 801)
ss = simp.CC(1j)*ww
hf = simp.lambdify(s, Vo, "numpy")
v = hf(ss)

plt.figure(figsize=(10,8))
plt.title("Magnitude Response of low pass filter")
plt.loglog(ww,abs(v), lw = 2, label = "Magnitude Response")
plt.ylabel(r"$|H(j\omega)|$")
plt.xlabel(r"$\omega$")
plt.grid()
plt.legend()
plt.show()



### Question 1

A,V,b=lowpass(R1, R2, C1, C2, G, 1/s)
Vo = V[3]
hs = sig.lti(*coeff(Vo))
t = np.linspace(0, 5e-3, 10**4)
t, v = sig.impulse(hs,T = t)

plt.figure(figsize=(10,8))
plt.title("Unit Step Response of a low pass filter.")
plt.plot(t, v, label = "step response")
plt.legend()
plt.ylabel("y(t)")
plt.xlabel("Time(s)")
plt.grid()
plt.show()

### Question 2

A,V,b = A,V,b=lowpass(R1, R2, C1, C2, G, Vi = 1)
Vo = V[3]
hs = sig.lti(*coeff(Vo))
t = np.linspace(0, 5e-3, 10**5)
t, v, _ = sig.lsim(hs, U = inSignal(t), T = t)

plt.figure(figsize=(10,8))
plt.title("Input and Output of a low pass filter")
plt.plot(t, inSignal(t), label="input")
plt.plot(t, v, "k", label = "input response")
plt.xlabel("t");plt.ylabel("output")
plt.legend()
plt.grid()
plt.show()

### Question 3

A,V,b=highpass(R1, R2, C1, C2, G, Vi)
print(f"G = {G}")
Vo = V[3]
print(Vo)
ww = np.logspace(0, 8, 801)
ss = simp.CC(1j)*ww
hf = simp.lambdify(s, Vo, "numpy")
v = hf(ss)

plt.figure(figsize=(10,8))
plt.title("Magnitude Response of high pass filter")
plt.loglog(ww,abs(v), lw = 2, label = "Magnitude Response")
plt.ylabel(r"$|H(j\omega)|$")
plt.xlabel(r"$\omega$")
plt.grid()
plt.legend()
plt.show()

### Question 4
A,V,b=highpass(R1, R2, C1, C2, G, Vi)
t = np.linspace(0, 1e-4, 1000)
hs = sig.lti(*coeff(Vo))
t, v, _ = sig.lsim(hs, U=indampedS(t), T = t)
plt.figure(figsize=(10,8))
plt.title("Response for high frequency")
plt.plot(t, v, label = "Damped response")
plt.plot(t, indampedS(t), label = "Input")
plt.legend()
plt.xlabel("t");plt.ylabel("output")
plt.grid()
plt.show()

t = np.linspace(0, 1, 1000)
hs = sig.lti(*coeff(Vo))
t, v, _ = sig.lsim(hs, U=indampedS(t, 5, 10), T = t)
plt.figure(figsize=(10,8))
plt.title("Response for low frequency")
plt.plot(t, v, label = "Damped response")
plt.plot(t, indampedS(t, 5, 10), label = "Input")
plt.legend()
plt.xlabel("t");plt.ylabel("output")
plt.grid()
plt.show()

### Question 5

A,V,b=highpass(R1, R2, C1, C2, G, 1/s)
Vo = V[3]
hs = sig.lti(*coeff(Vo))
t = np.linspace(0, 5e-3, 10**4)
t, v = sig.impulse(hs,T = t)

plt.figure(figsize=(10,8))
plt.title("Step Response of High Pass filter")
plt.plot(t, v, label = "step response")
plt.xlabel("t");plt.ylabel("output")
plt.legend()
plt.grid()
plt.show()