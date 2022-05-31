# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:03:36 2022

@author: Sasuke
"""
import numpy as np
import matplotlib.pyplot as plt
import numpy.fft as fft
from matplotlib import cm

# Example 1

t, dt = np.linspace(-np.pi,np.pi,64, False, retstep=True)
fmax=1/dt
y= np.sin(np.sqrt(2)*t)
y[0]=0 # the sample corresponding to -tmax should be set zero
y = fft.fftshift(y) # make y start with y(t=0)
Y = fft.fftshift(fft.fft(y))/64
w = np.linspace(-np.pi*fmax,np.pi*fmax,64, False)
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y))
plt.xlim([-10,10])
plt.ylabel(r"$|Y|$")
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)$")
plt.grid()
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),"ro")
plt.xlim([-10,10])
plt.ylabel(r"Phase of $Y$")
plt.xlabel(r"$\omega$")
plt.grid()
plt.show()

# Example 2

t1 = np.linspace(-np.pi,np.pi,64, False)
t2 = np.linspace(-3*np.pi,-np.pi,64, False)
t3 = np.linspace(np.pi,3*np.pi,64, False)
# y=sin(sqrt(2)*t)
plt.figure(2)
plt.plot(t1,np.sin(np.sqrt(2)*t1),"b")
plt.plot(t2,np.sin(np.sqrt(2)*t2),"r")
plt.plot(t3,np.sin(np.sqrt(2)*t3),"r")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.title(r"$\sin\left(\sqrt{2}t\right)$")
plt.grid()
plt.show()

# Example 3

t1 = np.linspace(-np.pi,np.pi,64, False)
t2 = np.linspace(-3*np.pi,-np.pi,64, False)
t3 = np.linspace(np.pi,3*np.pi,64, False)
y= np.sin(np.sqrt(2)*t)
plt.figure(3)
plt.plot(t1,y,"b")
plt.plot(t2,y,"r")
plt.plot(t3,y,"r")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.title(r"$\sin\left(\sqrt{2}t\right)$")
plt.grid()
plt.show()

# Example 4

t, dt=np.linspace(-np.pi,np.pi,64, False, retstep = True)
fmax=1/dt
y=t
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y = fft.fftshift(y) # make y start with y(t=0)
Y = fft.fftshift(fft.fft(y))/64.0
w = np.linspace(-np.pi*fmax,np.pi*fmax,64, False)
plt.figure()
plt.semilogx(abs(w),20*np.log10(abs(Y)),lw=2)
plt.xlim([1,10])
plt.ylim([-20,0])
plt.xticks([1,2,5,10],["1","2","5","10"])
plt.ylabel(r"$|Y|$ (dB)")
plt.title(r"Spectrum of a digital ramp")
plt.xlabel(r"$\omega$")
plt.grid()
plt.show()

# Example 5

t1 = np.linspace(-np.pi,np.pi,64, False)
t2 = np.linspace(-3*np.pi,-np.pi,64, False)
t3 = np.linspace(np.pi,3*np.pi,64, False)
n = np.arange(64)
wnd = fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/63))
y = np.sin(np.sqrt(2)*t1)*wnd
plt.figure(3)
plt.plot(t1,y,"bo")
plt.plot(t2,y,"ro")
plt.plot(t3,y,"ro")
plt.ylabel(r"$y$")
plt.xlabel(r"$t$")
plt.title(r"$\sin\left(\sqrt{2}t\right)\times w(t)$ with $t$ wrapping every $2\pi$ ")
plt.grid()
plt.show()

# Example 6

t, dt = np.linspace(-np.pi,np.pi,64, False, retstep = True)
fmax=1/dt
n = np.arange(64)
wnd = fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/63))
y = np.sin(np.sqrt(2)*t)*wnd
y[0]=0 # the sample corresponding to -tmax should be set zeroo
y = fft.fftshift(y) # make y start with y(t=0)
Y=fft.fftshift(fft.fft(y))/64
w = np.linspace(-np.pi*fmax,np.pi*fmax,64, False)
plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y))
plt.xlim([-8,8])
plt.ylabel(r"$|Y|$")
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
plt.grid()
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),"ro")
plt.xlim([-8,8])
plt.ylabel(r"Phase of $Y$")
plt.xlabel(r"$\omega$")
plt.grid()
plt.show()

# Example 7

t, dt = np.linspace(-4*np.pi,4*np.pi,256, False, retstep = True)
fmax=1/dt
n = np.arange(256)
wnd = fft.fftshift(0.54+0.46*np.cos(2*np.pi*n/256))
y = np.sin(np.sqrt(2)*t)
# y=sin(1.25*t)
y = y*wnd

y[0]=0 # the sample corresponding to -tmax should be set zeroo
y = fft.fftshift(y) # make y start with y(t=0)
Y = fft.fftshift(fft.fft(y))/256
w = np.linspace(-np.pi*fmax,np.pi*fmax,256, False)

plt.figure()
plt.subplot(2,1,1)
plt.plot(w,abs(Y),"b",w,abs(Y),"bo")
plt.xlim([-8,8])
plt.ylabel(r"$|Y|$")
plt.title(r"Spectrum of $\sin\left(\sqrt{2}t\right)\times w(t)$")
plt.grid()
plt.subplot(2,1,2)
plt.plot(w,np.angle(Y),"ro")
plt.xlim([-8,8])
plt.ylabel(r"Phase of $Y$")
plt.xlabel(r"$\omega$")
plt.grid()
plt.show()

# ENd of examples

# Helper function

def solve(f, lim, n, xlim, ylabel, ylabel_, xlabel,
          title, window = True, show = True, last = False, t2 = None):
    t, dt = np.linspace(-lim, lim, n, False, retstep = True)
    if last:
        t = t2;dt = t2[1] - t2[0]
    fmax = 1/dt
    y = f(t)
    if window:
        a = np.arange(n)
        wnd = fft.fftshift(0.54+0.46*np.cos(2*np.pi*a/n))
        y = y*wnd
    y[0] = 0
    y = fft.fftshift(y)
    Y = fft.fftshift(fft.fft(y))/n
    w = np.linspace(-np.pi*fmax, np.pi*fmax, n, False)
    
    if show:
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(w, abs(Y))
        plt.xlim([-xlim, xlim])
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid()
        plt.subplot(2,1,2)
        phase = np.angle(Y)
        phase[np.where(abs(Y)<3e-3)] = 0
        plt.plot(w, phase, "ro")
        plt.xlim([-xlim, xlim])
        plt.ylabel(ylabel_)
        plt.xlabel(xlabel)
        plt.grid()
        plt.show()
    
    return Y,w

def estimator(Y, w, window = 1, s = 1e-4):
    ii = np.where(w>0)
    ii_ = np.where(np.logical_and(abs(Y)>s, w>0))[0]
    np.sort(ii_)
    omega = sum(abs(Y[ii])**2*w[ii])/sum(abs(Y[ii])**2)
    delta = np.sum(np.angle(Y[ii_[1:window+1]]))/(window+1)
    print(f"Estimate Delta and Omega are {delta}, {omega} respectively.")
    
def cos(t, w = 1.5, d = 0.5):
    return np.cos(w*t + d)

def cc(t,w=0.86):
    return np.cos(w*t)**3

# Spectrum of cos^3(wt)

solve(cc, 4*np.pi, 256, 3, r"$|Y|$", "Phase of Y", r"$\omega$", 
      r"Spectrum of $cos^3(\omega t)$", False)
solve(cc, 4*np.pi, 256, 3, r"$|Y|$", "Phase of Y", r"$\omega$", 
      r"Spectrum of $cos^3(\omega t)$")

# cos(wt+delta)

Y, w = solve(cos, np.pi, 128, 3, r"$|Y|$", "Phase of Y", r"$\omega$", 
      r"Spectrum of $cos(\omega t + \delta)$")

estimator(Y,w)

# Noisy Cosine

def noisy(t,w0=1.5,delta=0.5):
    return np.cos(w0*t + delta) + 0.1*np.random.randn(128)

Y, w = solve(noisy, np.pi, 128, 3, r"$|Y|$", "Phase of Y", r"$\omega$", 
      r"Spectrum of noisy $cos(\omega t + \delta)$")

estimator(Y, w)

# Chirp

def chirp(t):
    return np.cos(16*(1.5+t/(2*np.pi))*t)
solve(chirp, np.pi, 1024, 60, r"$|Y|$", "Phase of Y", r"$\omega$", 
      "Spectrum of chirp function")
solve(chirp, np.pi, 1024, 60, r"$|Y|$", "Phase of Y", r"$\omega$", 
      "Spectrum of chirp function", False)

# Parts of chirp

t, dt = np.linspace(-np.pi, np.pi, 1024, False, retstep = True)
t_ = np.split(t, 16)
mag, phase = np.zeros((16,64)), np.zeros((16,64))

for i in range(len(t_)):
    Y, w = solve(chirp, 10, 64, 60, r"$|Y|$", "Phase of Y", r"$\omega$", 
          "Spectrum of chirp function",show = False,last = True, t2 = t_[i], window = False)
    mag[i] = abs(Y)
    phase[i] = np.angle(Y)

fig = plt.figure()
ax = plt.axes(projection = "3d")
fmax = 1/dt;t = t[::64]
w = np.linspace(-fmax*np.pi, fmax*np.pi, 64, False)
t, w = np.meshgrid(t, w)
s = ax.plot_surface(w,t, mag.T, cmap = cm.coolwarm,
                linewidth = 0, antialiased = False)
fig.colorbar(s, shrink=0.5, aspect = 5)
plt.ylabel("Frequency")
plt.xlabel("Time")
plt.show()
fig = plt.figure()
ax = plt.axes(projection = "3d")
s = ax.plot_surface(w,t, phase.T, cmap = cm.coolwarm,
                linewidth = 0, antialiased = False)
fig.colorbar(s, shrink=0.5, aspect = 5)
plt.ylabel("Frequency")
plt.xlabel("Time")

plt.show()
