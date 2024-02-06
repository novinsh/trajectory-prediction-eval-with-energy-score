#%%
# this file contains empirical results for
# Effect of sample size
# Sensitivty Study
#
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

SMALL_SIZE = 11
MEDIUM_SIZE = 12
BIGGER_SIZE = 14

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# import matplotlib as mpl
# mpl.rcParams.update(mpl.rcParamsDefault)


# generate ground truth
np.random.seed(0)
sample_size = 1000 # number of ground truth observations
x0 = np.zeros(sample_size) # initial step
sig = 0.2 # noise variance
c = [1.0, 1.0, 1.0] # autoregression coefficient
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


# visualize trajectories in 2D
plt.figure(figsize=(5,2))
for i in range(5):
    plt.plot(obs[i,0,:,0], obs[i,0,:,1]+i*0.5, '-o')
    plt.plot(obs[i,0,0,0], obs[i,0,0,1]+i*0.5, '-o', color='k')
plt.yticks([],[])
plt.ylabel('$s=2$ - all zeros')
plt.xlabel('$s=1$')
plt.grid(alpha=0.25)
# plt.xticks(range(4), range(1,5))
plt.ylim([-0.5,2.5])
plt.tight_layout()
plt.savefig("figs/synthetic_obs_trajectories.pdf", box_inches='tight')
plt.show()
plt.close()

plt.plot(obs[:,0,:,0].T, '.-', color='k', alpha=0.1,)
plt.legend()
plt.grid(alpha=0.25)
plt.xticks(range(obs.shape[2]), range(1,obs.shape[2]+1))
plt.xlabel('Temporal step l')
plt.ylabel('Values of the spatial dimension d=1')
plt.tight_layout()
plt.savefig("figs/synthetic_obs_trajectories_1d.pdf", box_inches='tight')
plt.show()

#%%

def generate_preds(mu_eps=[0,0,0], std_eps=[0, 0, 0], seed=0, visualize=True):
    """ generate predictions by purturbing the parameters of the ground truth model  """
    np.random.seed(seed)
    # eps1 = 0 
    # eps2 = 1.0
    traj_size = 300
    preds = []
    for i in range(sample_size):
        x0 = np.zeros(traj_size)
        x1 = np.random.normal(1+mu_eps[0], (1+std_eps[0])*sig, size=traj_size)
        x2 = np.random.normal(2+mu_eps[1], (2+std_eps[1])*sig, size=traj_size)
        x3 = np.random.normal(3+mu_eps[2], (3+std_eps[2])*sig, size=traj_size)
        y=np.zeros((traj_size,4,1))
        if i ==0 and visualize:
            plt.hist(x0)
            plt.hist(x1)
            plt.hist(x2, alpha=0.5)
            plt.hist(x3, alpha=0.5)
            plt.show()

        pred = np.concatenate((x0[:,np.newaxis], 
                               x1[:,np.newaxis], 
                               x2[:,np.newaxis],
                               x3[:,np.newaxis]), axis=-1) # (1000, 3)
        pred = np.concatenate((pred[...,np.newaxis], y), axis=2) # (1000, 3, 2)
        preds.append(pred)
    preds = np.array(preds)
    return preds

def generate_preds2(mu_eps=[0,0,0], std_eps=[0,0,0], sig=0.2, c=[1,1,1], seed=0, visualize=True):
    """ generate predictions by purturbing the parameters of the ground truth model """
    np.random.seed(seed)
    # eps1 = 0 
    # eps2 = 1.0
    traj_size = 300
    preds = []
    for i in range(sample_size):
        x0 = np.zeros(traj_size)
        x1 = c[0] * x0 + np.random.normal(1+mu_eps[0], sig+std_eps[0], size=traj_size)
        x2 = c[1] * x1 + np.random.normal(1+mu_eps[1], sig+std_eps[1], size=traj_size)
        x3 = c[2] * x2 + np.random.normal(1+mu_eps[2], sig+std_eps[2], size=traj_size)
        y=np.zeros((traj_size,4,1))

        pred = np.concatenate((x0[:,np.newaxis], 
                               x1[:,np.newaxis], 
                               x2[:,np.newaxis],
                               x3[:,np.newaxis]), axis=-1) # (1000, 3)
        pred = np.concatenate((pred[...,np.newaxis], y), axis=2) # (1000, 3, 2)
        preds.append(pred)

    preds = np.array(preds)
    return preds

c_preds = [1,1,1]
unbiased_preds = generate_preds2(mu_eps=[0, 0, 0], std_eps=[0, 0, 0], sig=sig, c=c)


multiplier = 1 # for vcisualization purposes NB: do not forget to set to 1 before running scoring calculations. 


biased_predictions = {
    'unbiased': generate_preds2(mu_eps=[0, 0, 0], std_eps=[0, 0, 0], sig=sig, c=c, seed=1),
    'symm_mean': generate_preds2(mu_eps=multiplier*np.array([0.01, 0.01, 0.01]), std_eps=[0, 0, 0], sig=sig, c=c_preds),
    'asymm_mean': generate_preds2(mu_eps=multiplier*np.array([0.005, -0.01, 0.015]), std_eps=[0, 0, 0], sig=sig, c=c_preds),
    'large_var': generate_preds2(mu_eps=[0, 0, 0], std_eps=multiplier*np.array([0.01, 0.01, 0.01]), sig=sig, c=c_preds),
    'small_var': generate_preds2(mu_eps=[0, 0, 0], std_eps=np.array([-0.035, -0.035, -0.035]) if multiplier ==1 else (multiplier/6)*np.array([-0.035, -0.035, -0.035]), sig=sig, c=c_preds),
    'symm_mean_nd_large_var': generate_preds2(mu_eps=multiplier*np.array([0.01, 0.01, 0.01]), std_eps=multiplier*np.array([0.01, 0.01, 0.01]), sig=sig, c=c_preds),
}


def visualize_prediction(bias_type, biased_preds, show_ground_truth, savefig=False):
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
    plt.xlabel('Temporal step $t$')
    plt.tight_layout()
    plt.savefig(f"figs/synthetic_preds_distribution_{bias_type}.pdf", box_inches='tight') if savefig else None
    plt.show()
    #
    plt.plot(unbiased_preds[idx,:100,:,0].T, '.-', color='b', alpha=0.1)
    plt.plot([], '.-', color='b', alpha=1, label='optimal prediction')
    plt.plot(biased_preds[idx,:100,:,0].T, '.-', color='r', alpha=0.1)
    plt.plot([], '.-', color='r', alpha=1, label='deviated prediction')
    plt.plot(unbiased_preds[idx,:,:,0].mean(axis=0), '.:', color='k', label='optimal prediction mean')
    plt.plot(biased_preds[idx,:,:,0].mean(axis=0), '.--', color='k', label='deviated prediction mean')
    if show_ground_truth:
        plt.plot(obs[idx,:,:,0].T, '.-', color='k', label='ground truth')
    plt.legend()
    plt.grid(alpha=0.25)
    plt.xticks(range(obs.shape[2]), range(1,obs.shape[2]+1))
    plt.xlabel('Temporal step $t$')
    plt.ylabel('Spatial dimension $s=1$')
    plt.tight_layout()
    plt.savefig(f"figs/synthetic_preds_trajectories_{bias_type}.pdf", box_inches='tight') if savefig else None
    plt.show()

name_of_predictions = [
    'unbiased'  ,
    'symm_mean' ,
    'asymm_mean',
    'large_var' ,
    'small_var' ,
    'symm_mean_nd_large_var'
]

visualize_prediction('unbiased', biased_predictions['unbiased'], show_ground_truth=False, savefig=True if multiplier >1 else False)
visualize_prediction('symm_mean', biased_predictions['symm_mean'], show_ground_truth=False, savefig=True if multiplier >1 else False)
visualize_prediction('asymm_mean', biased_predictions['asymm_mean'], show_ground_truth=False, savefig=True if multiplier >1 else False)
visualize_prediction('large_var', biased_predictions['large_var'], show_ground_truth=False, savefig=True if multiplier >1 else False)
visualize_prediction('small_var', biased_predictions['small_var'], show_ground_truth=False, savefig=True if multiplier >1 else False)
visualize_prediction('symm_mean_nd_large_var', biased_predictions['symm_mean_nd_large_var'], show_ground_truth=False, savefig=True if multiplier >1 else False)

# for i in range(20):
#     plt.plot(unbiased_preds[i,0,:,0], unbiased_preds[i,0,:,1]+i, '-o', color='b', alpha=0.5)
# plt.show()

# for i in range(20):
#     plt.plot(biased_preds[i,0,:,0], biased_preds[i,0,:,1]+i, '-o', color='r', alpha=0.5)
# plt.show()

#%%

def calculate_scores(preds, temporal_steps=[1,2,3,4], trajectory_sizes=[50,100,300]):
    horizons= pd.Index(temporal_steps, name='Temporal Step')
    sizes = trajectory_sizes
    sizes = [50, 100, 300]
    # sizes = [100]
    metrics = ["FDE-1", "FDE-10%", "ADE-1", "ADE-10%", "ES", "FES", "EST", "ESS"] # Add FESS
    index = pd.MultiIndex.from_product([metrics, sizes], names=['metric', 'size'])
    df = pd.DataFrame(columns=horizons, index=index, dtype=np.float64)

    pbar = tqdm(total=len(sizes)*len(horizons))
    for size in sizes:
        for horizon in horizons:
            ob = obs[:,:,:horizon]
            pred = preds[:,:size,:horizon]
            df.loc[("FDE-1", size), horizon] = fde_topN(ob, pred, 1)
            df.loc[("FDE-10%",size), horizon] = fde_topNPercent(ob, pred)
            #
            df.loc[("ADE-1", size), horizon] = ade_topN(ob, pred)
            df.loc[("ADE-10%", size), horizon] = ade_topNPercent(ob, pred)
            #
            df.loc[("ES", size), horizon] = energy_score(ob, pred, K=pred.shape[1]).mean()
            #
            df.loc[("EST", size), horizon] = energy_score_temporal(ob, pred, K=pred.shape[1]).mean()
            #
            df.loc[("ESS", size), horizon] = energy_score_spatial(ob, pred, K=pred.shape[1]).mean()
            #
            df.loc[("FES", size), horizon] = \
                energy_score(ob[:,:,-1][:,:,np.newaxis], pred[:,:,-1][:,:,np.newaxis], K=pred.shape[1]).mean()
            pbar.update(1)
    pbar.close()
    return df

df = calculate_scores(unbiased_preds, trajectory_sizes=[50, 100, 300]) 
(df*100).round(2)

#%%
display((df*100).round(1).loc[['FDE-1', 'FDE-10%', 'FES']])
print(
    (df*100).round(1).loc[['FDE-1', 'FDE-10%', 'FES']].to_latex(float_format="%.2f")
)

display((df*100).round(1).loc[['ADE-1', 'ADE-10%', 'ES']])
print(
    (df*100).round(1).loc[['ADE-1', 'ADE-10%', 'ES']].to_latex(float_format="%.2f")
)

display((df*100).round(1).loc[['EST', 'ESS']])
print(
    (df*100).round(1).loc[['EST', 'ESS']].to_latex(float_format="%.2f")
)

#%%
def zscore_to_pvalue(z):
    p_value = stats.norm.sf(abs(z))  # Calculate the p-value using the survival function (sf)
    return p_value

# Example usage
z_score = -1.1498
p_value = 2*zscore_to_pvalue(z_score)
print("Z-score:", z_score)
print("P-value:", p_value)
#%%
# Diebold-Mariano test

def dm_test(score_1, score_2):
    # Adopted from Dumas et al. (2022)
    """ The lower the p-value, indicating the higher difference between the two scores. 
        Null hypothesis (H0): Two sets of scores (score_1, score_2) are not different from each other.
        Alternative hypothesis (H1): Two sets of scores are different from each other.
        The test is two-tail and the p-value under 10% (equivalent to 5% for one-tail) is considered significant 
        to reject the H0. 
    """

    # compute test statistic
    loss_d = score_2-score_1  # shape = (n_days,)

    # Computing the loss differential size
    N = loss_d.size

    # Computing the test statistic
    mean_d = np.mean(loss_d)
    # print('mean_d %.4f' %(mean_d))
    var_d = np.var(loss_d, ddof=0) 
    std_d = np.std(loss_d, ddof=0)
    DM_stat = mean_d / (np.sqrt((1/N) * var_d))
    # DM_stat = mean_d / (std_d+1e-10)
    # DM_stat = (loss_d-mean_d) / std_d
    # print(np.sqrt((1/N) * var_d))
    # print(std_d)
    # print("#")
    # print('DM_stat %.4f' %(DM_stat))

    p_value = (1 - stats.norm.cdf(DM_stat))
    # p_value = 2*stats.norm.sf(abs(DM_stat))
    # print(DM_stat)
    # print(DM_stat.shape)
    # print(p_value)
    # print("#")
    # print(p_value.shape)

    return p_value

def plot_score_distribution(s0, s1, title=""):
    plt.title(title)
    plt.hist(s0)
    plt.hist(s1)
    plt.vlines(s0.mean(), 0, 10, color='blue')
    plt.vlines(s1.mean(), 0, 10, color='red')
    plt.show()

def calculate_sensitivity(biased_preds, temporal_steps=[1,2,3,4], trajectory_sizes=[100], plotting=False):
    horizons= pd.Index(temporal_steps, name='step') # 2, 3, 4
    # sizes = [50, 100, 300]
    sizes = trajectory_sizes
    metrics = ["FDE-1", "FDE-10%", "ADE-1", "ADE-10%", "ES", "FES", "EST", "ESS"] # Add FESS
    index = pd.MultiIndex.from_product([metrics, sizes], names=['metric', 'size'])
    df_dm = pd.DataFrame(columns=horizons, index=index, dtype=np.float64)
    #
    pbar = tqdm(total=len(sizes)*len(horizons))
    for size in sizes:
        for horizon in horizons:
            ob = obs[:,:,:horizon]
            pred = unbiased_preds[:,:size,:horizon]
            pred_biased = biased_preds[:,:size,:horizon]
            # print(ob.shape)
            # print(pred_biased.shape)
            # print(pred_biased.shape)

            #
            s0 =fde_topN(ob, pred, 1, axis=1) 
            s1 =fde_topN(ob, pred_biased, 1, axis=1) 
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="FDE Top1")
            df_dm.loc[("FDE-1", size), horizon] = dm_test(s0, s1)

            #
            s0 = fde_topNPercent(ob, pred, axis=1)
            s1 = fde_topNPercent(ob, pred_biased, axis=1)
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="FDE Top10%")
            df_dm.loc[("FDE-10%", size), horizon] = dm_test(s0, s1)

            #
            s0 = ade_topN(ob, pred, axis=1)
            s1 = ade_topN(ob, pred_biased, axis=1)
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="ADE Top1")
            df_dm.loc[("ADE-1", size), horizon] = dm_test(s0, s1) 

            #
            s0 = ade_topNPercent(ob, pred, axis=1)
            s1 = ade_topNPercent(ob, pred_biased, axis=1)
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="ADE Top10%")
            df_dm.loc[("ADE-10%", size), horizon] = dm_test(s0, s1) 

            #
            s0 = energy_score(ob, pred, K=pred.shape[1])
            s1 = energy_score(ob, pred_biased, K=pred.shape[1])
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="ES")
            df_dm.loc[("ES", size), horizon] = dm_test(s0, s1)

            #
            s0 = energy_score_temporal(ob, pred, K=pred.shape[1]).mean(axis=1)
            s1 = energy_score_temporal(ob, pred_biased, K=pred.shape[1]).mean(axis=1)
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="EST")
            df_dm.loc[("EST", size), horizon] = dm_test(s0, s1) 
            #
            s0 = energy_score_spatial(ob, pred, K=pred.shape[1]).mean(axis=1)
            s1 = energy_score_spatial(ob, pred_biased, K=pred.shape[1]).mean(axis=1)
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="ESS")
            df_dm.loc[("ESS", size), horizon] = dm_test(s0, s1) 
            #
            s0 = energy_score(ob[:,:,-1][:,:,np.newaxis], pred[:,:,-1][:,:,np.newaxis], K=pred.shape[1])
            s1 = energy_score(ob[:,:,-1][:,:,np.newaxis], pred_biased[:,:,-1][:,:,np.newaxis], K=pred_biased.shape[1])
            if size == 100 and horizon == 4 and plotting:
                plot_score_distribution(s0, s1, title="FES")
            df_dm.loc[("FES", size), horizon] = dm_test(s0, s1)
            pbar.update(1)
    pbar.close()
    return df_dm

plotting = False
df_dm_1 = calculate_sensitivity(biased_predictions["unbiased"], temporal_steps=[4], trajectory_sizes=[100], plotting=plotting)
print("unbiased")
display((df_dm_1*100).droplevel(1).round(2))

df_dm_2 = calculate_sensitivity(biased_predictions["symm_mean"], temporal_steps=[4], trajectory_sizes=[100], plotting=plotting)
print("symm_mean")
display((df_dm_2*100).droplevel(1).round(2))

df_dm_3 = calculate_sensitivity(biased_predictions["asymm_mean"], temporal_steps=[4], trajectory_sizes=[100], plotting=plotting)
print("asymm_mean")
display((df_dm_3*100).droplevel(1).round(2))

df_dm_4 = calculate_sensitivity(biased_predictions["large_var"], temporal_steps=[4], trajectory_sizes=[100], plotting=plotting)
print("large_var")
display((df_dm_4*100).droplevel(1).round(2))

df_dm_5 = calculate_sensitivity(biased_predictions["small_var"], temporal_steps=[4], trajectory_sizes=[100], plotting=plotting)
print("small_var")
display((df_dm_5*100).droplevel(1).round(2))

# df_dm_6 = calculate_sensitivity(biased_predictions['symm_mean_nd_large_var'], temporal_steps=[4], trajectory_sizes=[100], plotting=plotting)
# print("symm_mean_nd_large_var")
# display((df_dm_6*100).droplevel(1).round(2))

#%%
df_final = (pd.concat([df_dm_1, df_dm_2, df_dm_3, df_dm_4, df_dm_5], axis=1)*100).droplevel(1).round(1)
df_final.columns = pd.Index([1,2,3,4,5], name='Cases')
df_final.index = pd.Index(['$FDE_{(L=1)}$', '$FDE_{(L=10)}$', '$ADE_{(L=1)}$', '$ADE_{(L=10)}$', '$ES$', '$FES$', '$EST$', '$ESS$'], name='Metric')
display(df_final)
print(df_final.to_latex(float_format="%.1f"))
# (df_dm*100).droplevel(1).round(2)#.style.highlight_max(props='textbf:--rwrap;')
#%%
#%%
# print( 
#     (df_dm*100).droplevel(1).round(2).to_latex() 
# ) 

# print(
#     (df_dm*100).droplevel(1).round(2).loc[['FDE top1', 'FDE top10%', 'ADE top1', 'ADE top10%',]].to_latex()
# )

# print(
#     (df_dm*100).droplevel(1).round(2).loc[['ES', 'FES', 'EST', 'ESS']].to_latex()
# )

