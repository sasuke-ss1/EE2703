# -*- coding: utf-8 -*-
"""
Created on Sun Mar  6 11:48:33 2022

@author: Sasuke
"""

from pylab import *
import mpl_toolkits.mplot3d.axes3d as p3 
import argparse
import numpy as np

try:
    parser = argparse.ArgumentParser(description = "Takes inputs")
    parser.add_argument("Nx", metavar = "Nx", nargs = "?", 
                        const = 1, default=25, type = int)
    parser.add_argument("Ny", metavar = "Ny", nargs = "?", 
                        const = 1, default=25, type = int)
    parser.add_argument("Radius", metavar = "R", nargs = "?", 
                        const = 1, default=8, type = float)
    parser.add_argument("Niter", metavar = "Niter", nargs = "?", 
                        const = 1, default=1500, type = int)
    
    args = parser.parse_args()

 
except:
    parser = argparse.ArgumentParser(description = "Takes inputs")
    parser.add_argument("-Nx", metavar = "--Nx", default = 25, type = int)
    parser.add_argument("-Ny", metavar = "--Ny", default = 25, type = int)
    parser.add_argument("-Radius", metavar = "--R", default = 8, type = float)
    parser.add_argument("-Niter", metavar = "--Niter", default = 1500, type = int)
    
    args = parser.parse_args()
    
Nx, Ny, Radius, Niter = args.Nx, args.Ny, args.Radius, args.Niter

phi = np.zeros((Ny, Nx))
x = np.linspace(-0.5, 0.5, Nx);y =np.linspace(-0.5, 0.5, Ny)
X,Y= meshgrid(x, y)
radius = 1.01*Radius/(min(Nx, Ny) -1) #Tolerance of 1% in radius
print(radius) 
ii = np.where(X ** 2 + Y ** 2 <= radius** 2)
phi[ii] = 1.0

# Plotting Countours
figure(figsize=(10, 8))
title("Contour Plot of the Potential", fontsize=16)
clabel(plt.contour(X, Y, phi))
xlabel("$x$", fontsize=16)
ylabel("$y$", fontsize=16)
scatter(X[0, ii[1]], Y[ii[0], 0], color="r", label="V=1")
legend()
grid()
show()

#Interations
error = np.zeros((Niter, ))
for k in range(Niter):
    oldphi = phi.copy()
    phi[1:-1,1:-1]=0.25*(oldphi[1:-1,0:-2]+ oldphi[1:-1,2:]+ 
                         oldphi[:-2,1:-1] + oldphi[2:,1:-1])
    
    phi[1:-1, 0], phi[1:-1, -1], phi[-1, :] = phi[1:-1, 1], phi[1:-1, -2], phi[-2, :]
    phi[ii] = 1.0
    error[k] = abs(phi-oldphi).max()

#loglog and semilog plot    
figure(figsize = (70,70))
subplot(10,10,1)
title("Error vs Niter in loglog", fontsize = 16)
loglog(np.arange(1,Niter + 1), error, label = "error")
grid()
xlabel("Niter", fontsize=16)
ylabel("Absolute Error", fontsize=16)
legend()
subplot(10,10,2)
title("Error vs Niter in semilog", fontsize = 16)
semilogy(np.arange(1,Niter + 1), error, label = "error")
grid()
legend()
xlabel("Niter", fontsize=16)
ylabel("Absolute Error", fontsize=16)
show()

#Plotting 50th point
figure(figsize=(10, 8))
title("Error vs Niter", fontsize=16)
xlabel("Niter", fontsize=16)
ylabel("Absolute Error", fontsize=16)
e = error[::50]
semilogy(np.arange(1, len(e) + 1), e, "ro",  label = "Error")
grid()
legend()
show()

#fitting
A = np.concatenate([np.arange(1,Niter+1).reshape(-1,1),np.ones((Niter,1))]
                   , axis =1)
fit1 = np.linalg.lstsq(A, np.log(error).reshape(-1,1), rcond=-1)[0]
#For pointer after 500 iterations
fit2 = np.linalg.lstsq(A[500:],
                       np.log(error)[500:].reshape(-1,1), rcond=-1)[0]


#plotting fitted lines
figure(figsize=(70,70))
x_ = np.arange(1,Niter+1)
subplot(10,10,1)
title("loglog plot with fited lines", fontsize = 16)
loglog(np.arange(1,Niter + 1), error, label = "Actual")
loglog(x_, np.exp(fit1[1] + fit1[0]*x_), label = "Full fit")
loglog(x_, np.exp(fit2[1] + fit2[0]*x_), label = "Fit after 500 iter.")
legend()
grid()

subplot(10,10,2)
title("Semilog plot with fited lines", fontsize = 16)
semilogy(x_, error, label = "Actual")
semilogy(x_, np.exp(fit1[1] + fit1[0]*x_), label = "Full fit")
semilogy(x_, np.exp(fit2[1] + fit2[0]*x_), label = "Fit after 500 iter.")
legend()
grid()
show()

#Cummulative sum
figure(figsize=(70,70))
subplot(10,10,1)
title("Cummulative Sum in loglog", fontsize = 16)
loglog(x_, -np.exp(fit2[1])/fit2[0]*np.exp(fit2[0]*(x_+0.5)), label = "Error")
grid()
xlabel("Niter", fontsize=16)
ylabel("Error", fontsize=16)
legend()
subplot(10,10,2)
title("Cummulative Sum in semilog", fontsize = 16)
semilogy(x_, -np.exp(fit2[1])/fit2[0]*np.exp(fit2[0]*(x_+0.5)), label = "Error")
grid()
xlabel("Niter", fontsize=16)
ylabel("Error", fontsize=16)
legend()
show()

#3D plot
fig = plt.figure(figsize=(10,8))
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, phi,cmap = cm.jet,linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
title('3-D Surface plot of the potential')
xlabel('x')
ylabel('y')
show()

#Contour potential plot
figure(figsize=(10, 8))
title("Contour Plot of the Potential", fontsize=16)
clabel(plt.contour(X, Y, phi))
xlabel("$x$", fontsize=16)
ylabel("$y$", fontsize=16)
scatter(X[0, ii[1]], Y[ii[0], 0], color="r", label="V=1")
legend()
grid()
show()

#Quiver Plot
Jx = np.zeros_like(phi)
Jy = np.zeros_like(phi)
Jx[:,1:-1] = (phi[:,:-2] - phi[:,2:])/2
Jy[1:-1,:] = (phi[:-2,:] - phi[2:,:])/2

figure(figsize=(10, 8))
title("The vector plot of current flow", fontsize=16)
scatter(X[0, ii[1]], Y[ii[0], 0], color="r", label="$\phi=1$")
quiver(X, Y, Jx[:,::1], Jy[:,::1], scale=4)
xlabel("x", fontsize=16)
ylabel("y", fontsize=16)
legend()
show()

    
        