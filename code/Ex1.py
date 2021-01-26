import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd
import sobol_new as sn
from pathlib import Path as P
import utils

np.random.seed(999)

plt.style.use("seaborn-white")

# =============================================================================
# Parameters
# =============================================================================

m = 256 # Number of steps
r = 0.5 # Risk-free rate
sigma = 0.3 # Volatility
T = 2 # Time to maturity
S0 = 5 # Initial stock price
K = 10 # Strike

t = np.linspace(0,T,m) # Time grid

# Simulations
n_simulations = 10000

# Confidence intervals
alpha = 0.05
crit_val = st.t.ppf(1-alpha/2, n_simulations-1)

kappa = 20 # K splits for RQMC

# =============================================================================
# Compute \mathbb{E}(\Psi) for P_1
# =============================================================================
# MC 
muMC, stdMC, seMC = utils.runSimMC(m, r, sigma, t, T, S0, n_simulations,
                             K, utils.asian, par=None, payoff="p1")

# QMC

muQMC, stdQMC, seQMC = utils.runSimQMC(m, r, sigma, t, T, S0, n_simulations, K,
                                 kappa, utils.asian, par=None, payoff="p1")

res_df = pd.DataFrame({"mu":[muMC, muQMC],
                        "CI":[crit_val*seMC, crit_val*seQMC],
                        "SE":[seMC, seQMC]})

res_df.index = ["CMC", "RQMC"]
print(res_df)
print(res_df.round(4).to_latex())

# # =============================================================================
# # Compute \mathbb{E}(\Psi) for P_2 and different values of \beta
# # =============================================================================

BETA = [1, 2, 5, 10, 20]

res_df = {"muMC":[],"ciMC":[], "seMC":[],
          "muQMC":[],"ciQMC":[], "seQMC":[]}

for i in range(len(BETA)):

    muMC, stdMC, seMC = utils.runSimMC(m, r, sigma, t, T, S0, n_simulations,
                              K, utils.asian, par=BETA[i], payoff="p2")
    
    muQMC, stdQMC, seQMC = utils.runSimQMC(m, r, sigma, t, T, S0, n_simulations, K,
                                  kappa, utils.asian, par=BETA[i], payoff="p2")
    
    ciMC = crit_val*seMC
    ciQMC = crit_val*seQMC
    
    res_df["muMC"].append(muMC)
    res_df["ciMC"].append(ciMC)
    res_df["seMC"].append(seMC)
    
    res_df["muQMC"].append(muQMC)
    res_df["ciQMC"].append(ciQMC)
    res_df["seQMC"].append(seQMC)
    
res_df = pd.DataFrame(res_df)
res_df.index = BETA
print(res_df)
print(res_df.round(4).to_latex())

# =============================================================================
# Generate plots for $\Psi_1$
# =============================================================================

N = np.logspace(1, 5, 30).astype(np.int)

res_df = {"muMC":[], "seMC":[],
          "muQMC":[], "seQMC":[]}

for n in N:
    muMC, stdMC, seMC = utils.runSimMC(m, r, sigma, t, T, S0, n,
                              K, utils.asian, par=None, payoff="p1")
    
    muQMC, stdQMC, seQMC = utils.runSimQMC(m, r, sigma, t, T, S0, n, K,
                                  kappa, utils.asian, par=None, payoff="p1")
    
    res_df["muMC"].append(muMC)
    res_df["seMC"].append(seMC)
    res_df["muQMC"].append(muQMC)
    res_df["seQMC"].append(seQMC)

res_df = pd.DataFrame(res_df)

# Error as function of N

plt.plot(N, res_df["seMC"], label="CMC", linestyle="--", color="grey")
plt.plot(N, res_df["seQMC"], label="RQMC", color="black")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.xlabel("N")
plt.ylabel("Standard Error")
plt.savefig("figures/Ex1/Ex1_SEvsN_psi1.pdf")
plt.show()

# mu as function of N

plt.plot(N, res_df["muMC"], label="CMC", linestyle="--", color="grey")
plt.plot(N, res_df["muQMC"], label="RQMC", color="black")
plt.legend()
plt.xscale("log")
plt.xlabel("N")
plt.ylabel(r"$\mathbb{E}(\Psi)$")
plt.savefig("figures/Ex1/Ex1_MUvsN_psi1.pdf")
plt.show()
    
# =============================================================================
# Generate plots for $\Psi_2$
# =============================================================================
BETA = [1, 2, 5, 10, 20]
N = np.logspace(1, 5, 30).astype(np.int)

for i in range(len(BETA)): #Loop \beta

    res_df = {"muMC":[], "seMC":[],
          "muQMC":[], "seQMC":[]}
    for n in N: #Loop N
        muMC, stdMC, seMC = utils.runSimMC(m, r, sigma, t, T, S0, n,
                              K, utils.asian, par=BETA[i], payoff="p2")
    
        muQMC, stdQMC, seQMC = utils.runSimQMC(m, r, sigma, t, T, S0, n, K,
                                      kappa, utils.asian, par=BETA[i], payoff="p2")
        
        res_df["muMC"].append(muMC)
        res_df["seMC"].append(seMC)
        res_df["muQMC"].append(muQMC)
        res_df["seQMC"].append(seQMC)
        
    res_df = pd.DataFrame(res_df)
    
    # Error as function of N

    plt.plot(N, res_df["seMC"], label="CMC", linestyle="--", color="grey")
    plt.plot(N, res_df["seQMC"], label="RQMC", color="black")
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("N")
    plt.ylabel("Standard Error")
    plt.title(r"$\beta$"+f" = {BETA[i]}")
    plt.savefig(f"figures/Ex1/Ex1_SEvsN_psi2_beta{BETA[i]}.pdf")
    plt.show()
    
    # mu as function of N

    plt.plot(N, res_df["muMC"], label="CMC", linestyle="--", color="grey")
    plt.plot(N, res_df["muQMC"], label="RQMC", color="black")
    plt.legend()
    plt.xscale("log")
    plt.xlabel("N")
    plt.ylabel(r"$\mathbb{E}(\Psi)$")
    plt.title(r"$\beta$"+f" = {BETA[i]}")
    plt.savefig(f"figures/Ex1/Ex1_MUvsN_psi2_beta{BETA[i]}.pdf")
    plt.show()
