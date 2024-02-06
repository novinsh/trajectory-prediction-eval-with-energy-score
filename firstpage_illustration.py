#%%
import numpy as np
from metrics import fde_topN, energy_score
import seaborn as sns
import matplotlib.pyplot as plt
import scipy
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

N=2000
K=100

np.random.seed(0)
gt = np.random.normal(0,1, size=N) # ground truth
f0 = np.random.normal(0,1, size=N*K) # optimal
f1 = np.random.normal(0.1,1.5, size=N*K) # suboptimal

sns.kdeplot(gt.ravel()[:1000], color='b', fill=True, label='Ground truth:\n $\mathcal{N}(0,1)$')
sns.kdeplot(f1.ravel()[:4000], color='r', fill=True, label='Forecast:\n $\mathcal{N}(0.1,1.5)$')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)
plt.grid(alpha=0.25)
plt.savefig("figs/first_illustration/unimodal.pdf",bbox_inches='tight')
plt.show()


# data samples, distribution samples, temporal, spatial
fde_f0 = fde_topN(gt.reshape(N,1,1,1), f0.reshape(N,K,1,1))
fde_f1 = fde_topN(gt.reshape(N,1,1,1), f1.reshape(N,K,1,1))

es_f0 = energy_score(gt.reshape(N,1,1,1), f0.reshape(N,K,1,1))
es_f1 = energy_score(gt.reshape(N,1,1,1), f1.reshape(N,K,1,1))
print(fde_f0.round(3))
print(fde_f1.round(3))

print(es_f0.mean().round(3))
print(es_f1.mean().round(3))


#%%

def mixture_gaussian(norm_params, weights=[0.8, 0.2], size=N):
    # n_components = norm_params.shape[0]
    # weights = np.array()
    mixture_idx = np.random.choice(len(weights), size=size, replace=True, p=weights)
    return np.fromiter((scipy.stats.norm.rvs(*(norm_params[i])) for i in mixture_idx), dtype=np.float64)

gt_params = np.array([[0, 0.5],
                     [3, 0.5]])
gt_m = mixture_gaussian(norm_params=gt_params, weights=[0.8, 0.2], size=N)
f0_m = mixture_gaussian(norm_params=gt_params, weights=[0.8, 0.2], size=N*K)

f1_params = np.array([[0, 0.6],
                     [3, 0.6]])
f1_m = mixture_gaussian(norm_params=f1_params, weights=[0.7, 0.3], size=N*K)

np.random.seed(0)
sns.kdeplot(gt_m.ravel()[:1000], color='b', fill=True, label='Ground truth:\n $0.8\mathcal{N}(0,0.5)+0.2\mathcal{N}(3,0.5)$')
sns.kdeplot(f1_m.ravel()[:2000], color='r', fill=True, label='Forecast:\n $0.7\mathcal{N}(0,0.6)+0.3\mathcal{N}(3,0.6)$')
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=2)
plt.grid(alpha=0.25)
plt.savefig("figs/first_illustration/bimodal.pdf",bbox_inches='tight')
plt.show()

# data samples, distribution samples, temporal, spatial
fde_f0 = fde_topN(gt_m.reshape(N,1,1,1), f0_m.reshape(N,K,1,1))
fde_f1 = fde_topN(gt_m.reshape(N,1,1,1), f1_m.reshape(N,K,1,1))

es_f0 = energy_score(gt_m.reshape(N,1,1,1), f0_m.reshape(N,K,1,1))
es_f1 = energy_score(gt_m.reshape(N,1,1,1), f1_m.reshape(N,K,1,1))
print(fde_f0.round(3))
print(fde_f1.round(3))

print(es_f0.mean().round(3))
print(es_f1.mean().round(3))

#%%
fig, axes = plt.subplots(1,2, figsize=(12,4),sharey=True)
plt.subplots_adjust(wspace=0.1)
sns.kdeplot(gt.ravel()[:], color='b', fill=True, label='Ground truth: $\mathcal{N}(0,1)$', ax=axes[0])
sns.kdeplot(f1.ravel()[:], color='r', fill=True, label='Forecast: $\mathcal{N}(0.1,1.5)$', ax=axes[0])
axes[0].axvline(gt.mean(), color='b', linestyle='--', linewidth=2)
axes[0].axvline(f1.mean(), color='r', linestyle='--', linewidth=2)
axes[0].legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=1)
axes[0].grid(alpha=0.2)
#
sns.kdeplot(gt_m.ravel()[:], color='b', fill=True, label='Ground truth: $0.8\mathcal{N}(0,0.5)+0.2\mathcal{N}(3,0.5)$', ax=axes[1])
sns.kdeplot(f1_m.ravel()[:], color='r', fill=True, label='Forecast: $0.7\mathcal{N}(0,0.6)+0.3\mathcal{N}(3,0.6)$', ax=axes[1])
axes[1].legend(loc='lower left', bbox_to_anchor=(0, 1.05), ncol=1)
axes[1].grid(alpha=0.2)
plt.savefig("figs/first_illustration/uni-and-bi-modal.pdf",bbox_inches='tight')
plt.show()