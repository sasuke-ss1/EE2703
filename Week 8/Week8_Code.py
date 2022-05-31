# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 11:29:51 2022

@author: Sasuke
"""

import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft

def gauss(x):
    return np.exp(-0.5*x**2)

def expectedfn(w):
    return 1/np.sqrt(2*np.pi) * np.exp(-w**2/2)

### Question 1

# Random data
x = np.random.rand(100)
X = fft.fft(x)
y = fft.ifft(X)
print(abs(x-y).max())

# Sepctrum of sin(5t) no shift
x = np.linspace(0, 2*np.pi, 128)
y = np.sin(5*x)
Y = fft.fft(y)
plt.figure()
plt.subplot(2,1,1)
plt.title(r"Spectrum of $sin(5t)$ without shifting")
plt.plot(abs(Y), lw = 2, label = "Magnitude")
plt.xlabel("k");plt.ylabel(r"$|Y|$")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.xlabel("k");plt.ylabel("Phase of Y")
plt.plot(np.unwrap(np.angle(Y)), lw = 2)
plt.grid()
plt.show()

# Sepctrum of sin(5t) with shift

x = np.linspace(0, 2*np.pi, 128, endpoint = False)
y = np.sin(5*x)
Y = fft.fftshift(fft.fft(y))/128
w = np.linspace(-64,63,128)
plt.figure()
plt.subplot(2,1,1)
plt.title(r"Spectrum of $sin(5t)$")
plt.plot(w,abs(Y), lw = 2, label = "Magnitude")
plt.xlim([-10,10])
plt.xlabel("k");plt.ylabel(r"$|Y|$")
plt.grid();plt.legend()
plt.subplot(2,1,2)
ii = np.where(abs(Y)>1e-3)
plt.xlabel("k");plt.ylabel("Phase of Y")
plt.plot(w, np.angle(Y), "ro", lw = 2)
plt.plot(w[ii], np.angle(Y[ii]), "go", lw = 2)
plt.xlim([-10,10])
plt.grid()
plt.show()

# Spectrum of (1+0.1cos(t))cos(10t)
t = np.linspace(0, 2*np.pi, 128, endpoint = False)
y = (1+0.1*np.cos(t))*np.cos(10*t)
Y = fft.fftshift(fft.fft(y))/128
w = np.linspace(-64,63,128)
plt.figure()
plt.subplot(2,1,1)
plt.title(r"Spectrum of $(1+0.1cos(t))cos(10*t)$")
plt.plot(w,abs(Y), lw = 2, label = "Magnitude")
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.xlabel(r"$\omega$");plt.ylabel("Phase of Y")
plt.plot(w, np.angle(Y), "ro", lw = 2)
plt.xlim([-15,15])
plt.grid()
plt.show()

# Spectrum of (1+0.1cos(t))cos(10t) with more points

t = np.linspace(-4*np.pi, 4*np.pi, 512, endpoint = False)
y = (1+0.1*np.cos(t))*np.cos(10*t)
Y = fft.fftshift(fft.fft(y))/512
w = np.linspace(-64,63,512, False)
plt.figure()
plt.subplot(2,1,1)
plt.title(r"Spectrum of $(1+0.1cos(t))cos(10*t)$")
plt.plot(w,abs(Y), lw = 2, label = "Magnitude")
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.xlabel(r"$\omega$");plt.ylabel("Phase of Y")
plt.plot(w, np.angle(Y), "ro", lw = 2)
plt.xlim([-15,15])
plt.grid()
plt.show()

### Question 2

# Spectrum of sin^3(t)

t = np.linspace(-4*np.pi, 4*np.pi, 512, endpoint = False)
y = np.sin(t)**3
Y = fft.fftshift(fft.fft(y))/512
w = np.linspace(-64,63,512, False)
plt.figure()
plt.subplot(2,1,1)
plt.title(r"Spectrum of $sin^3(t)$")
plt.plot(w,abs(Y), lw = 2, label = "Magnitude")
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.xlabel(r"$\omega$");plt.ylabel("Phase of Y")
ii=np.where(abs(Y)>1e-3)
plt.plot(w[ii], np.angle(Y[ii]), "go", lw = 2)
plt.xlim([-15,15])
plt.grid()
plt.show()

# Spectrum of cos^3(t)

t = np.linspace(-4*np.pi, 4*np.pi, 512, endpoint = False)
y = np.cos(t)**3
Y = fft.fftshift(fft.fft(y))/512
w = np.linspace(-64,63,512, False)
plt.figure()
plt.subplot(2,1,1)
plt.title(r"Spectrum of $cos^3(t)$")
plt.plot(w,abs(Y), lw = 2, label = "Magnitude")
plt.xlim([-15,15])
plt.ylabel(r"$|Y|$")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.xlabel(r"$\omega$");plt.ylabel("Phase of Y")
ii=np.where(abs(Y)>1e-3)
plt.plot(w[ii], np.angle(Y[ii]), "go", lw = 2)
plt.xlim([-15,15])
plt.grid()
plt.show()

### Question 3

t = np.linspace(-4*np.pi, 4*np.pi, 512, endpoint = False)
y = np.cos(20*t + 5*np.cos(t))
Y = fft.fftshift(fft.fft(y))/512
w = np.linspace(-64,63,512, False)
plt.figure()
plt.subplot(2,1,1)
plt.title(r"Spectrum of $cos(20t + 5cos(t))$")
plt.plot(w,abs(Y), lw = 2, label = "Magnitude")
plt.xlim([-30,30])
plt.ylabel(r"$|Y|$")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.xlabel(r"$\omega$");plt.ylabel("Phase of Y")
ii=np.where(abs(Y)>1e-3)
plt.plot(w[ii], np.angle(Y[ii]), "go", lw = 2)
plt.xlim([-30,30])
plt.grid()
plt.show()

### Question 4

Yold = 0;iters=0;N=128;T=8*np.pi
err = 1e-6+1
while err>1e-6:
    x = np.linspace(-T/2, T/2, N, False)
    w = np.linspace(-N*np.pi/T, N*np.pi/T, N, False)
    y = gauss(x)
    Y=fft.fftshift(fft.fft(fft.ifftshift(y)))*T/(2*np.pi*N)
    err = sum(abs(Y[::2]-Yold))
    Yold = Y
    iters+=1
    T*=2
    N*=2

true_err = sum(abs(Y-expectedfn(w)))
print(f"True Error = {true_err}")

mag = abs(Y)
phi = np.angle(Y)
phi[np.where(mag<1e-6)]=0
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y), label = "Magnitude")
plt.xlim([-5,5])
plt.ylabel('Magnitude')
plt.title("Estimate fft of gaussian")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro')
ii=np.where(abs(Y)>1e-3)
plt.plot(w[ii],np.angle(Y[ii]),'go')
plt.xlim([-5,5])
plt.ylabel("Phase")
plt.xlabel(r"$\omega$")
plt.grid()
plt.show()

# Plotting expected output
    
Y_ = expectedfn(w)

mag = abs(Y_)
phi = np.angle(Y_)
phi[np.where(mag<1e-6)]=0
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y), label = "Magnitude")
plt.xlim([-5,5])
plt.ylabel('Magnitude')
plt.title("True fft of gaussian")
plt.grid();plt.legend()
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),'ro')
ii=np.where(abs(Y)>1e-3)
plt.plot(w[ii],np.angle(Y[ii]),'go')
plt.xlim([-5,5])
plt.xlabel(r"$\omega$")
plt.ylabel("Phase")
plt.grid()
plt.show()




