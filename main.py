import numpy as np
from numpy import pi, sin, cos, sqrt
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import warnings

L = 2*pi  #[nm]
h = L/1e3
v0 = 8
hbar = 1
m = 1
#z = np.arange(L, 1.9*L, h)
z0 = L/(2*hbar) * np.sqrt(2*m*v0)

#Definerer funksjonene
def funk1(z): return np.tan(z) - np.sqrt(z0**2/z**2 - 1)
def funk2(z): return np.tan(z+pi/2) - np.sqrt(z0**2/z**2 - 1)
def energi(z): return (2*hbar**2*z**2)/(m*L**2) -v0


#Plotter løsningene av ligning f mhp. z
def plotKryss(f, x):
    warnings.filterwarnings('ignore')
    z = np.arange(0, z0, .01)
    y = f(z)
    y[:-1][np.diff(y) < 0] = np.nan
    plt.plot(x, np.zeros(len(x)), 'x', markersize=7)
    plt.plot(z, y)

    intervall = np.arange(0, z0, pi)
    vertikal = [-20, 20]
    for i in intervall:
        plt.plot([i, i], vertikal, '--', color='black', linewidth='.8')
    plt.grid(linestyle='--')
    plt.ylim(-20, 20)


#Plotter energinivåene
def plotEnergi(energi):
    x = np.array([0., 10.])
    for e in energi:
        plt.plot(x, [e, e])
    plt.xticks([])


#Finner løsningene av funk1 og funk2 mtp. z
init_gues1 = np.array([1.4, 4.4, 7.3, 10.3])
z1 = fsolve(funk1, init_gues1)
e1 = energi(z1)

init_gues2 = np.array([2.4, 5.4, 8.3, 11.3])
z2 = fsolve(funk2, init_gues2)
e2 = energi(z2)


# Dette er for å dele plotten inn i intervaller med bare ét nullpunkt
x = np.arange(0, z0, pi/2)
y = [-20, 20]


#Plotter løsningene for z og energinivåene
def mkfig_op1():
    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plotKryss(funk1, z1)
    plt.title("Løsninger med tan(z)")
    plt.subplot(122)
    plotEnergi(e1)
    plt.title("Energi")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.subplot(121)
    plotKryss(funk2, z2)
    plt.title("Løsninger med tan(z + pi/2)")
    plt.subplot(122)
    plotEnergi(e2)
    plt.title("Energi")
    plt.show()




mkfig_op1()

'''
#Vet ikke om dette trengs
def wave_out(x, E):
    a, b = 1, 1
    k = sqrt(2*m*E)/hbar
    return a*sin(k*x) + b*cos(k*x)

def wave_in(x, E):
    c, d = 1, 1
    k = sqrt(2*m*(v0 + E))/hbar
    return c*sin(k*x) + d*cos(k*x)
'''

