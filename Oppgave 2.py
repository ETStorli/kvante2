import numpy as np
from numpy import pi, sin, cos, sqrt
import scipy.constants as sc
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

hbar = 1
m = 1
Dx = .001
w = 1

def potensial(x): return 0.5*m*w**2*x**2

#n = odde, gammel, trengs ikke
def psi_odde(E, V, L):
    x = np.arange(0, L, Dx)
    psiH = np.zeros(len(x))
    psiH[0] = 0
    psiH[1] = 1
    for i in range(len(x)-2):
        psiH[i+2] = -2*m*Dx**2/hbar**2*(E-V(x[i]))*psiH[i+1] + 2*psiH[i+1] - psiH[i]
        if abs((psiH[i+2]-psiH[i+1])/2) > 0.01:
            x = x[:i+2]
            psiH = psiH[:i+2]
            break
    #Pga. symmetri, og for å ikke ta med 'psi0' to ganger:
    psiV = -psiH[-1:0:-1]
    psi = np.append(psiV, psiH)
    x = np.append(-x[-1:0:-1], x)
    return x, psi

#n = jevn, gammel, trengs ikke
def psi_jevn(E, V, L):
    x = np.arange(0, L, Dx)
    psiH = np.zeros(len(x))
    psiH[0] = 1
    psiH[1] = (-m*Dx**2/hbar**2*(E-V(x[0]))+1)*psiH[0]
    for i in range(len(x)-2):
        psiH[i+2] = -2*m*Dx**2/hbar**2*(E-V(x[i]))*psiH[i+1] + 2*psiH[i+1] - psiH[i]
        if abs((psiH[i+2]-psiH[i+1])/2) > 0.01:
            x = x[:i+2]
            psiH = psiH[:i+2]
            break
    #Pga. symmetri, og for å ikke ta med 'psi0' to ganger:
    psiV = psiH[-1:0:-1]
    psi = np.append(psiV, psiH)
    x = np.append(-x[-1:0:-1], x)
    return x, psi


def nullpunkt(x, psi):
    Nullpunkt = []
    for i in range(len(x)-1):
        if psi[i] == 0:
            Nullpunkt.append(x[i])
        elif psi[i]*psi[i+1] <0:        #Bruker det at funksjonen skifter fortegn når den krysser x-aksen
            Nullpunkt.append(x[i])
    return Nullpunkt[1:-1]      #Vil ikke ha med det første eller siste nullpunktet siden de stammer fra feil


def Shooting(n, V, L, tol):
    x = np.arange(0, L, Dx)
    psiH = np.zeros(len(x))
    E = .5 + n
    if n%2 == 1:
        psiH[0] = 0
        psiH[1] = 1
        a = -1      #Pga. antisymmetrisk
    else:
        psiH[0] = 1
        psiH[1] = (-m * Dx ** 2 / hbar ** 2 * (E - V(x[0])) + 1) * psiH[0]
        a = 1       #Pga symmetrisk
    for i in range(len(x)-2):
        print("dcfgbhj")

        psiH[i+2] = -2*m*Dx**2/hbar**2*(E-V(x[i]))*psiH[i+1] + 2*psiH[i+1] - psiH[i]
        if abs((psiH[i+2]-psiH[i+1])/2) > tol:
            x = x[:i+2]
            psiH = psiH[:i+2]
            break
    psiV = a*psiH[-1:0:-1]
    psi = np.append(psiV, psiH)
    print(psi)
    x = np.append(-x[-1:0:-1], x)

    Nullpunkt = nullpunkt(x, psi)
    plt.plot(x, psi)
    plt.plot(Nullpunkt, np.zeros(len(Nullpunkt)), 'x', markersize=7)
    plt.grid()
    plt.show()
    print("x_i = ", x)
    print("psi_i = ", psi)
    print("Antall nullpunkter: ", len(Nullpunkt))
    print(Nullpunkt)

Shooting(0, potensial, 15, .01)