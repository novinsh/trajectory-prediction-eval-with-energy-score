#%%
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from metrics import fde_topNPercent, fde_topN

import matplotlib as mpl

# Set global parameters for matplotlib
mpl.rcParams['axes.titlesize'] = 18  # Title size
mpl.rcParams['axes.labelsize'] = 16  # Label size
mpl.rcParams['xtick.labelsize'] = 14  # X-axis tick size
mpl.rcParams['ytick.labelsize'] = 14  # Y-axis tick size
mpl.rcParams['font.size'] = 13       # Global font size

# Set line width for plot curves
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 8  # Marker size
mpl.rcParams['lines.markeredgewidth'] = 2  # Marker width
#%%

from functools import partial
def bold_formatter(x, value, num_decimals=2):
    # https://github.com/pandas-dev/pandas/issues/38328
    if round(x, num_decimals) == round(value, num_decimals):
        return f'\\textbf{{{x:.{num_decimals}f}}}' 
    else:
        return f'{x:.{num_decimals}f}' 

elements = np.array([0, 1])
probs = [0.9, 0.1] # ground truth
probs_pred_unb = probs # unbiased
probs_pred_uniform = [0.5, 0.5]  # random/uniform - underconfidence
probs_pred_biased = [0.1, 0.9] # overconfident in the wrong class
probs_pred_biased2 = [0.999, 0.001] # overconfident in the right class
probs_pred_biased3 = [0.8, 0.2] # underconfident in the right class

plot_fig = True 
save_fig = False

if plot_fig:
    plt.figure(figsize=(5,5))
    plt.stem(elements, probs,                    orientation='vertical', linefmt='b-', markerfmt='bo', basefmt='k-', label=f'$p_y={probs[1]}$ (ground truth)')
    plt.stem(elements+0.025, probs_pred_unb,     orientation='vertical', linefmt='c-', markerfmt='co', basefmt='k-', label=f'$p_x={probs_pred_unb[1]}$ (optimal)')
    plt.stem(elements+0.05, probs_pred_uniform,  orientation='vertical', linefmt='g-', markerfmt='go', basefmt='k-', label=f'$p_x={probs_pred_uniform[1]}$ (uniform)')
    plt.stem(elements+0.075, probs_pred_biased,  orientation='vertical', linefmt='r-', markerfmt='ro', basefmt='k-', label=f'$p_x={probs_pred_biased[1]}$')
    plt.stem(elements+0.1, probs_pred_biased2,   orientation='vertical', linefmt='m-', markerfmt='mo', basefmt='k-', label=f'$p_x={probs_pred_biased2[1]}$')
    plt.stem(elements+0.125, probs_pred_biased3, orientation='vertical',  linefmt='y-', markerfmt='yo', basefmt='k-', label=f'$p_x={probs_pred_biased3[1]}$')
    plt.xticks(elements, elements)
    plt.yticks(np.unique(probs+
                        probs_pred_unb+
                        probs_pred_uniform+
                        probs_pred_biased+
                        probs_pred_biased2+
                        probs_pred_biased3), rotation=0)    
    plt.ylim([-0.01,1.01])
    # plt.ticklabel_format(style='plain', axis='y')
    plt.xlabel("Outcome")
    plt.ylabel("Probability")
    plt.grid(0.25)
    plt.legend(loc="lower center")
    # plt.yscale('log')
    plt.savefig("figs/bernoulli_simulation.pdf", dpi=300, bbox_inches='tight') if save_fig else None
    plt.show()

np.random.seed(0)

n_obs = 10000
n_samples = 1000 # number of scenarios or trajectories
percnt = 0.2

# (num_samples x num_scenarios x num_leadtime x num_spatial_dims)
obs =       np.random.choice(elements, (n_obs,1,1,1), p=probs)
pred_u =    np.random.choice(elements, (n_obs,n_samples,1,1), p=probs_pred_unb)  
pred_uni =  np.random.choice(elements, (n_obs,n_samples,1,1), p=probs_pred_uniform) 
pred_b =    np.random.choice(elements, (n_obs,n_samples,1,1), p=probs_pred_biased) 
pred_b2 =   np.random.choice(elements, (n_obs,n_samples,1,1), p=probs_pred_biased2) 
pred_b3 =   np.random.choice(elements, (n_obs,n_samples,1,1), p=probs_pred_biased3) 

err_u = fde_topNPercent(obs, pred_u, percnt)
err_uni = fde_topNPercent(obs, pred_uni, percnt)
err_b = fde_topNPercent(obs, pred_b, percnt)
err_b2 = fde_topNPercent(obs, pred_b2, percnt)
err_b3 = fde_topNPercent(obs, pred_b3, percnt)

print("minimum of N")
print(f"true    {err_u:2.5f}")
print(f"uniform {err_uni:2.5f}")
print(f"biased1 {err_b:2.5f}")
print(f"biased2 {err_b2:2.5f}")
print(f"biased3 {err_b3:2.5f}")

minDE = [err_u, err_uni, err_b, err_b2, err_b3]

err_u   = fde_topN(obs, pred_u, N=n_samples)
err_uni = fde_topN(obs, pred_uni, N=n_samples)
err_b   = fde_topN(obs, pred_b, N=n_samples)
err_b2  = fde_topN(obs, pred_b2, N=n_samples)
err_b3  = fde_topN(obs, pred_b3, N=n_samples)

DE = [err_u, err_uni, err_b, err_b2, err_b3]

print("\nAverage error")
print(f"true    {err_u:2.5f}")
print(f"uniform {err_uni:2.5f}")
print(f"biased1 {err_b:2.5f}")
print(f"biased2 {err_b2:2.5f}")
print(f"biased3 {err_b3:2.5f}")

errors = np.stack([minDE, DE], axis=1)
df_results = pd.DataFrame(errors, columns=["minDE", "DE"], 
                          index=[f"$p_x={probs_pred_unb[1]}$ (optimal)", 
                                 f"$p_x={probs_pred_uniform[1]}$ (uniform)", 
                                 f"$p_x={probs_pred_biased[1]}$", 
                                 f"$p_x={probs_pred_biased2[1]}$", 
                                 f"$p_x={probs_pred_biased3[1]}$"])
fmts_min_3f = {column: partial(bold_formatter, value=df_results[column].min(), num_decimals=3) for column in df_results.columns}
print(df_results.to_latex(float_format="%.3f", formatters=fmts_min_3f, escape=False))
df_results
#%%

# from copy import deepcopy

# n_obs = 10000
# n_samples = 1000 # number of scenarios or trajectories
percnts = [0.01, 0.1, 0.2, 0.5, 0.8, 0.9, 0.99,] # equal to the probability of the true class
minDEs = []

for percnt in percnts:
    err_u = fde_topNPercent(obs, pred_u, percnt)
    err_uni = fde_topNPercent(obs, pred_uni, percnt)
    err_b = fde_topNPercent(obs, pred_b, percnt)
    err_b2 = fde_topNPercent(obs, pred_b2, percnt)
    err_b3 = fde_topNPercent(obs, pred_b3, percnt)
    # print([err_u, err_uni, err_b, err_b2, err_b3])
    minDEs.append([err_u, err_uni, err_b, err_b2, err_b3])

minDEs = np.array(minDEs)
DE = np.array(DE)
errors = np.append(minDEs, DE[np.newaxis,:], axis=0).T
minDE_colnames = [f"minDE($\\alpha$={a})" for a in percnts]
df_results = pd.DataFrame(errors, 
                          columns=minDE_colnames+["DE"], 
                          index=[f"$p_x={probs_pred_unb[1]}$ (optimal)", 
                                 f"$p_x={probs_pred_uniform[1]}$ (uniform)", 
                                 f"$p_x={probs_pred_biased[1]}$", 
                                 f"$p_x={probs_pred_biased2[1]}$", 
                                 f"$p_x={probs_pred_biased3[1]}$"])
                                 
fmts_min_3f = {column: partial(bold_formatter, value=df_results[column].min(), num_decimals=3) for column in df_results.columns}
print(df_results.to_latex(float_format="%.3f", formatters=fmts_min_3f, escape=False))
df_results.style.apply(lambda col: ['font-weight:bold' if x==col.min() else '' for x in col])