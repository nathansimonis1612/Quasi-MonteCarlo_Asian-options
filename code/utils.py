import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import sobol_new as sn

def simGBM(m, r, sigma, t, T, S0, n_simulations, dW):
    """Generate path for stock price"""
    dt = T/m
    W = np.cumsum(dW*np.sqrt(dt), axis=1)
    t = np.broadcast_to(t, shape=(n_simulations,m))
    X = (r-0.5*sigma**2)*t + sigma*W
    S = S0 * np.exp(X)
    S[:,0] = S0
    return S

def runSimMC(m, r, sigma, t, T, S0, n_simulations,
             K, func, par=None, payoff="p1"):
    """Run simulations for crude MC"""
    dW = np.random.normal(size=(n_simulations, m))
    S = simGBM(m, r, sigma, t, T, S0, n_simulations, dW)
    zeta = np.mean(S, axis=1) - K
    psi = func(zeta, par, payoff)

    muMC = np.mean(psi)
    stdMC = np.std(psi)
    seMC = stdMC/np.sqrt(n_simulations)

    return muMC, stdMC, seMC


def runSimQMC(m, r, sigma, t, T, S0, n_simulations,
              K, kappa, func, par=None, payoff="p1"):
    """Run simulations for Randomly shifted QMC"""
    n_simulations = int(np.ceil(n_simulations / kappa))
    sobol_seq = sn.generate_points(n_simulations, m)
    MU = np.zeros(kappa)
    
    for i in range(kappa):        
        U = np.random.uniform(size=(m,))   
        U = np.broadcast_to(U, shape=(n_simulations, m))

        shifted_seq = np.add(sobol_seq, U) % 1
        
        dW = st.norm.ppf(shifted_seq)

        S = simGBM(m, r, sigma, t, T, S0, n_simulations, dW)
        zeta = np.mean(S, axis=1) - K
        psi = func(zeta, par, payoff)
        MU[i] = np.mean(psi)

    muQMC = np.mean(MU)
    stdQMC = np.std(MU)
    seQMC = stdQMC / np.sqrt(kappa)

    return muQMC, stdQMC, seQMC


def asian(zeta, beta = None, payoff = "p1"):
    """Payoff functions for asian options"""
    if payoff == "p1":
        return np.maximum(zeta, 0)
    elif payoff == "p2":  
        return np.log(1+np.exp(beta*zeta))/beta
    
def binary_asian(zeta, gamma = None, payoff = "p1"):
    """Payoff functions for binary options"""
    if payoff == "p1":
        return 20 * (1.0*(zeta>0))
    elif payoff == "p2":
        return 20*(1+np.exp(-2*gamma*zeta))**(-1)
