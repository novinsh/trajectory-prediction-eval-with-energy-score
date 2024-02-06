#%%
# this file contains the emprical results for the propriety showcase

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from metrics import (fde_topN, ade_topN, 
                              fde_topNPercent, ade_topNPercent, 
                              energy_score, 
                              energy_score_temporal, energy_score_spatial,
                              )
from tqdm.auto import tqdm
from scipy import stats
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

# generate ground truth
np.random.seed(0)
sample_size = 1000 # number of ground truth observations
x0 = np.zeros(sample_size)
sig = 0.2
# c = [1.0, 0, 1]
c = [1.0, 1.0, 1.0]
x1 = c[0] * x0 + np.random.normal(1, sig, size=sample_size)
x2 = c[1] * x1 + np.random.normal(1, sig, size=sample_size)
x3 = c[2] * x2 + np.random.normal(1, sig, size=sample_size)

print(x0.shape)
print(x1.shape)
print(x2.shape)
print(x3.shape)

# visualize distribution
plt.hist(x0, density=True, label='$y_{11}$')
plt.hist(x1, density=True, label='$y_{21}$')
plt.hist(x2, alpha=0.5, density=True, label='$y_{31}$')
plt.hist(x3, alpha=0.5, density=True, label='$y_{41}$')
plt.xlabel('Temporal step (l)')
plt.ylabel('Density')
plt.grid(alpha=0.25)
plt.legend()
plt.tight_layout()
plt.savefig("figs/synthetic_obs_distribution.pdf", bbox_inches='tight')
plt.show()
plt.close()

# put together the ground truth trajectories
y=np.zeros((sample_size,4,1))
obs = np.concatenate((x0[:,np.newaxis], 
                               x1[:,np.newaxis], 
                               x2[:,np.newaxis],
                               x3[:,np.newaxis]), axis=-1) # (1000, 3)
obs = np.concatenate((obs[...,np.newaxis], y), axis=2) # (1000, 3, 2)
obs = obs[:,np.newaxis]

#%%

def generate_preds2(mu_eps=[0,0,0], std_eps=[0,0,0], sig=0.2, c_eps=[1,1,1], seed=0, visualize=True):
    np.random.seed(seed)
    # eps1 = 0 
    # eps2 = 1.0
    traj_size = 500
    preds = []
    for i in range(sample_size):
        x0 = np.zeros(traj_size)
        x1 = (c[0]+c_eps[0]) * x0 + np.random.normal(1+mu_eps[0], sig+std_eps[0], size=traj_size)
        x2 = (c[1]+c_eps[1]) * x1 + np.random.normal(1+mu_eps[1], sig+std_eps[1], size=traj_size)
        x3 = (c[2]+c_eps[2]) * x2 + np.random.normal(1+mu_eps[2], sig+std_eps[2], size=traj_size)
        y=np.zeros((traj_size,4,1))

        pred = np.concatenate((x0[:,np.newaxis], 
                               x1[:,np.newaxis], 
                               x2[:,np.newaxis],
                               x3[:,np.newaxis]), axis=-1) # (1000, 3)
        pred = np.concatenate((pred[...,np.newaxis], y), axis=2) # (1000, 3, 2)
        preds.append(pred)

    preds = np.array(preds)
    return preds

c_preds = [0,0,0]
unbiased_preds = generate_preds2(mu_eps=[0, 0, 0], std_eps=[0, 0, 0], sig=sig, c_eps=[0,0,0])
#
# uncomment to choose a certain biased/suboptimal prediction
# bias_type = 'symm_mean'
bias_type = 'asymm_mean'
# bias_type = 'var'
# bias_type = 'symm_mean_nd_large_var'
# bias_type = 'correlation'
# bias_type = 'asymm_var'
multiplier = 1 # for vcisualization purposes
# plt.plot(np.linspace(0,1,20)**3, np.zeros(20), 'x')
grid_size=7

if bias_type == "correlation":
    deviations = np.linspace(-2,1,19) 
elif bias_type == "symm_mean" or bias_type == "asymm_mean":
    # deviations = np.linspace(-0.05,0.05,19) 
    deviations = np.arange(-0.05, 0.05, 0.005).round(3)[1:]
else:
    dd = np.linspace(0,1,grid_size)**1.5
    scale = 0.05
    d_scaled = scale * dd 
    # plt.plot(d_scaled, np.zeros(10), 'x')
    d = np.concatenate([-d_scaled[1:], d_scaled])
    d.sort()
    print(d)
    # plt.plot(d, np.zeros(2*grid_size-1), 'x')
    # plt.show()
    deviations = d
    # deviations = np.arange(-0.1, 0.11, 0.02).round(2)
    deviations = np.arange(-0.1, 0.1, 0.005).round(3)
    # deviations = np.arange(-0.08, 0.081, 0.01).round(3)
    # deviations = np.append(deviations,[0])
    deviations.sort()
print(deviations)
print(len(deviations))
biased_predictions = []
show_ground_truth = False
if bias_type == 'symm_mean':
    # biased_preds = generate_preds2(mu_eps=multiplier*np.array([0.025, 0.025, 0.025]), std_eps=[0, 0, 0], sig=sig, c_eps=c_preds) # symmetric mean bias 
    for b in deviations:
        biased_preds = generate_preds2(mu_eps=multiplier*np.array([b, b, b]), std_eps=[0, 0, 0], sig=sig, c_eps=c_preds) # large variance bias
        biased_predictions.append(biased_preds)
elif bias_type == 'asymm_mean':
    for b in deviations:
        biased_preds = generate_preds2(mu_eps=multiplier*np.array([b, -b, b]), std_eps=[0, 0, 0], sig=sig, c_eps=c_preds) # large variance bias
        biased_predictions.append(biased_preds)
elif bias_type == 'var':
    # biased_preds = generate_preds2(mu_eps=[0, 0, 0], std_eps=multiplier*np.array([0.05, 0.05, 0.05]), sig=sig, c_eps=c_preds) # large variance bias
    for b in deviations:
        biased_preds = generate_preds2(mu_eps=[0, 0, 0], std_eps=multiplier*np.array([0, 0, b]), sig=sig, c_eps=c_preds) # large variance bias
        biased_predictions.append(biased_preds)
elif bias_type == 'symm_mean_nd_large_var':
    biased_preds = generate_preds2(mu_eps=multiplier*np.array([0.025, 0.025, 0.025]), std_eps=multiplier*np.array([0.05, 0.05, 0.05]), sig=sig, c_eps=c_preds) # mean and variance bias
elif bias_type == 'correlation':
    for b in deviations:
        biased_preds = generate_preds2(mu_eps=multiplier*np.array([0.0, 0.0, 0.0]), std_eps=multiplier*np.array([0.0, 0.0, 0.0]), sig=sig, c_eps=np.array([b, -b, b])) # mean and variance bias
        biased_predictions.append(biased_preds)
elif bias_type == 'asymm_var':
    for b in deviations:
        biased_preds = generate_preds2(mu_eps=[0, 0, 0], std_eps=multiplier*np.array([b, 0, -b]), sig=sig, c_eps=c_preds) # large variance bias
        biased_predictions.append(biased_preds)


idx=0 # sample idx
#
fig, axes = plt.subplots(2,1,figsize=(10,7), sharex=True,)
shape = unbiased_preds.shape
axes[0].set_title('Unbiased Prediction')
axes[0].hist(unbiased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,0,0], label='$x_{11}$', density=True)
axes[0].hist(unbiased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,1,0], label='$x_{21}$', density=True)
axes[0].hist(unbiased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,2,0], label='$x_{31}$', alpha=0.75, density=True)
axes[0].hist(unbiased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,3,0], label='$x_{41}$', alpha=0.75, density=True)
axes[0].grid(alpha=0.5)
axes[0].set_ylim([0,3])
axes[0].set_ylabel('Density')
axes[0].legend()
axes[1].set_title('Biased Prediction')
axes[1].hist(biased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,0,0], label='$x_{11}$', density=True)
axes[1].hist(biased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,1,0], label='$x_{21}$', density=True)
axes[1].hist(biased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,2,0], label='$x_{31}$', alpha=0.75, density=True)
axes[1].hist(biased_preds.reshape((shape[0]*shape[1],)+shape[2:])[:,3,0], label='$x_{41}$', alpha=0.75, density=True)
axes[1].grid(alpha=0.5)
axes[1].set_ylim([0,3])
axes[1].set_ylabel('Density')
axes[1].legend()
plt.xlabel('Temporal step (l)')
plt.tight_layout()
plt.savefig(f"figs/propriety/synthetic_preds_distribution_{bias_type}.pdf", box_inches='tight')
plt.show()
#
plt.plot(unbiased_preds[idx,:100,:,0].T, '.-', color='b', alpha=0.1)
plt.plot([], '.-', color='b', alpha=1, label='unbiased prediction')
plt.plot(biased_preds[idx,:100,:,0].T, '.-', color='r', alpha=0.1)
plt.plot([], '.-', color='r', alpha=1, label='biased prediction')
plt.plot(unbiased_preds[idx,:,:,0].mean(axis=0), '.:', color='k', label='unbiased prediction mean')
plt.plot(biased_preds[idx,:,:,0].mean(axis=0), '.--', color='k', label='biased prediction mean')
if show_ground_truth:
    plt.plot(obs[idx,:,:,0].T, '.-', color='k', label='ground truth')
plt.legend()
plt.grid(alpha=0.25)
plt.xticks(range(obs.shape[2]), range(1,obs.shape[2]+1))
plt.xlabel('Temporal step l')
plt.ylabel('Values of the spatial dimension d=1')
plt.tight_layout()
plt.savefig(f"figs/propriety/synthetic_preds_trajectories_{bias_type}.pdf", box_inches='tight')
plt.show()


#%%

# trajectories.reshape(1000,1,[np.newaxis] # samples, scenarios, leadtime, spatial dim
# horizons= pd.Index([1, 2, 3, 4], name='Temporal Step')
horizons= pd.Index([4], name='Temporal Step')
sizes = [50, 100, 300, 500]
# sizes = [50, 100, 300]
biased_preds_id = [f"b{i}" for i, _ in enumerate(biased_predictions)]
pred_types = ["unb",] + biased_preds_id
print(pred_types)
metrics = ["FDE top1", "FDE top10%", "FDE", "ADE top1", "ADE top10%", "ADE", "ES", "FES", "EST", "ESS"] # Add FESS
index = pd.MultiIndex.from_product([metrics, pred_types, sizes], names=['metric', 'pred', 'size'])
df = pd.DataFrame(columns=horizons, index=index, dtype=np.float64)
df

pbar = tqdm(total=len(pred_types)*len(sizes)*len(horizons))
for pred_type, preds in zip(pred_types, [unbiased_preds]+ biased_predictions):
    for size in sizes:
        for horizon in horizons:
            ob = obs[:,:,:horizon]
            pred = preds[:,:size,:horizon]
            df.loc[("FDE top1", pred_type, size), horizon] = fde_topN(ob, pred, 1)
            df.loc[("FDE top10%", pred_type, size), horizon] = fde_topNPercent(ob, pred)
            df.loc[("FDE", pred_type, size), horizon] = fde_topN(ob, pred, size)
            
            df.loc[("ADE top1", pred_type, size), horizon] = ade_topN(ob, pred)
            df.loc[("ADE top10%", pred_type, size), horizon] = ade_topNPercent(ob, pred)
            df.loc[("ADE", pred_type, size), horizon] = ade_topN(ob, pred, size)
            #
            df.loc[("ES", pred_type, size), horizon] = energy_score(ob, pred, K=pred.shape[1]).mean()
            
            df.loc[("EST", pred_type, size), horizon] = energy_score_temporal(ob, pred, K=pred.shape[1]).mean()
            #
            df.loc[("ESS", pred_type, size), horizon] = energy_score_spatial(ob, pred, K=pred.shape[1]).mean()
            #
            df.loc[("FES", pred_type, size), horizon] = \
                energy_score(ob[:,:,-1][:,:,np.newaxis], pred[:,:,-1][:,:,np.newaxis], K=pred.shape[1]).mean()
            pbar.update(1)
pbar.close()

(df*100).round(2)
#%%
metrics_to_plot = ['FDE top1', 'FDE top10%', 'FDE', 'ADE top1', 'ADE top10%', 'ADE', 'ES', 'EST', 'ESS']
step=4
unbiased_idx = np.where(deviations == 0)[0][0]

for metric_name in metrics_to_plot:
    plt.title("minFDE" if metric_name == "FDE top1" else metric_name)
    # plt.xlabel(f"Parameter Deviation ({bias_type})")
    plt.xlabel(f"Parameter Deviation")
    plt.ylabel(f"Score")
    plt.plot(unbiased_idx, df.loc[(metric_name, "unb", 50), step], 'x', color='blue')
    plt.plot(unbiased_idx, df.loc[(metric_name, "unb", 100), step], 'x', color='green')
    plt.plot(unbiased_idx, df.loc[(metric_name, "unb", 300), step], 'x', color='red')
    plt.plot(unbiased_idx, df.loc[(metric_name, "unb", 500), step], 'x', color='orange')
    scores_1 = df.loc[[(metric_name, id, 50) for id in biased_preds_id], step]
    scores_2 = df.loc[[(metric_name, id, 100) for id in biased_preds_id], step]
    scores_3 = df.loc[[(metric_name, id, 300) for id in biased_preds_id], step]
    scores_4 = df.loc[[(metric_name, id, 500) for id in biased_preds_id], step]
    plt.plot(scores_1.values, color='blue')
    plt.plot(scores_2.values, color='green')
    plt.plot(scores_3.values, color='red')
    plt.plot(scores_4.values, color='orange')
    plt.plot(scores_1.argmin(), scores_1.iloc[scores_1.argmin()], 'o', color='blue')
    plt.plot(scores_2.argmin(), scores_2.iloc[scores_2.argmin()], 'o', color='green')
    plt.plot(scores_3.argmin(), scores_3.iloc[scores_3.argmin()], 'o', color='red')
    plt.plot(scores_4.argmin(), scores_4.iloc[scores_4.argmin()], 'o', color='orange')
    plt.plot([], color='blue', linestyle='-', label='predictions ($K=50$)')
    plt.plot([], color='green', linestyle='-', label='predictions ($K=100$)')
    plt.plot([], color='red', linestyle='-', label='predictions ($K=300$)')
    plt.plot([], color='orange', linestyle='-', label='scores (500 trajectories)')
    plt.scatter([], [], marker='x', color='k', label='ground truth\'s score')
    plt.scatter([], [], marker='o', color='k', label='minimum score')
    plt.xticks(range(len(deviations)), deviations.round(3), rotation=90)
    plt.grid(alpha=0.1)
    # plt.ylim([0,0.5])
    plt.legend(loc='upper left')
    mn = metric_name.replace("%", "p")
    # plt.savefig(f"figs/propriety/{mn}_{bias_type}.pdf", bbox_inches='tight', tight_layout=True)
    plt.show()
    plt.close()

#%%

print( 
    (df*100).round(2).to_latex() 
) 

print(
    (df*100).round(1).loc[['FDE top1', 'FDE top10%', 'FES']].to_latex()
)

print(
    (df*100).round(1).loc[['ADE top1', 'ADE top10%', 'ES', 'EST']].to_latex()
)

