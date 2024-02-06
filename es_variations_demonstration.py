#%%
import numpy as np
import matplotlib.pyplot as plt
from metrics import (energy_score, 
                              energy_score_spatial, 
                              energy_score_temporal
                              )


#%%
# 1D spatial example
# same spatial but different temporal
forecast_a = np.array([[0,0], [1,1]])
forecast_b = np.array([[0,1], [1,0]])
groundtruth = forecast_a

# same temporal but different spatial
forecast_a = np.array([[0,0], [1,1]])
forecast_b = np.array([[0,1], [1,0]])
groundtruth = forecast_a

obs_i = np.random.choice([0,1], size=10, p=[0.5, 0.5])

observations = groundtruth[obs_i]

fig, axes = plt.subplots(1,2, figsize=(10,5), sharey=True)
axes[0].set_title('forecast A')
axes[0].plot(forecast_a.T, 'x-')
axes[0].grid(alpha=0.25)
axes[0].set_aspect('equal', 'box')
axes[0].set_xlabel('T')
axes[0].set_ylabel('S=1')
axes[0].set_xticks(range(2), range(2))
axes[1].set_title('forecast B')
axes[1].plot(forecast_b.T, 'x-')
axes[1].grid(alpha=0.25)
axes[1].set_aspect('equal', 'box')
axes[1].set_xlabel('T')
axes[1].set_ylabel('S=1')
axes[1].set_xticks(range(2), range(2))
plt.show()


est_as = []
est_bs = []
ess_as = []
ess_bs = []
es_as = []
es_bs = []

for obs in observations:
    # conform to the dimensionality for my energy score implementation 
    # n_sample,s n_scenarios, n_horizon, n_dims
    y = obs.reshape(1, 1, 2, 1).astype('float')
    # forecasts are constant for new obs
    x_a = forecast_a.reshape(1, 2, 2, 1).astype('float') 
    x_b = forecast_b.reshape(1, 2, 2, 1).astype('float')
    d = np.product(y.shape[2:])
    # n_samples = x.shape[0]
    # n_scenarios = x.shape[1]
    # n_horizon = x.shape[2]
    # n_dims = x.shape[3]
    est_a = energy_score_temporal(y, x_a, K=x_a.shape[1], d=d)
    est_b = energy_score_temporal(y, x_b, K=x_b.shape[1], d=d)
    ess_a = energy_score_spatial(y, x_a, K=x_a.shape[1], d=d)
    ess_b = energy_score_spatial(y, x_b, K=x_b.shape[1], d=d)
    es_a = energy_score(y, x_a, K=x_a.shape[1], d=d)
    es_b = energy_score(y, x_b, K=x_b.shape[1], d=d)
    est_as.append(est_a.ravel())
    est_bs.append(est_b.ravel())
    ess_as.append(ess_a.ravel())
    ess_bs.append(ess_b.ravel())
    es_as.append(es_a.ravel())
    es_bs.append(es_b.ravel())
    # assert False

plt.figure(figsize=(4,2))
plt.bar(0, np.mean(est_as))
plt.bar(1, np.mean(est_bs))
plt.bar(2, np.mean(ess_as))
plt.bar(3, np.mean(ess_bs))
plt.bar(4, np.mean(es_as))
plt.bar(5, np.mean(es_bs))
plt.xticks([0, 1, 2, 3, 4, 5], ['EST a', 'EST b', 'ESS a', 'ESS b', 'ES a', 'ES b'])
plt.grid(alpha=0.25 )
plt.show()
# print(forecast_a.shape)
# print(forecast_a)

print(f"EST a: {np.mean(est_as):2.3f}")
print(f"EST b: {np.mean(est_bs):2.3f}")
print(f"ESS a: {np.mean(ess_as):2.3f}")
print(f"ESS b: {np.mean(ess_bs):2.3f}")
print(f"ES  a: {np.mean(es_as):2.3f}")
print(f"ES  b: {np.mean(es_bs):2.3f}")

#%%
# 2D spatial example
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


# example with temporal score different, spatial score the same 
toyexample=1
if toyexample==1:
    forecast_a = np.array([[[0,0],[1,0]], 
                        [[0,1],[1,1]]])


    forecast_b = np.array([[[0,0],[1,1]], 
                        [[0,1],[1,0]]])

    # forecast_b = np.array([[[0,0],[1,2]], 
    #                        [[0,2],[1,0]]])

    groundtruth = forecast_a 

    # example with temporal score the same, spatial score different
    # forecast_a = np.array([[[0,0],[0,0]], 
    #                        [[1,1],[0,0]]])


    # forecast_b = np.array([[[0,1],[0,0]], 
    #                        [[1,0],[0,0]]])


    # groundtruth = forecast_a 
elif toyexample == 2:
    # example 2
    forecast_a = np.array([[[0,0],[0.5,0.5]], 
                        [[0.5,0.5],[1,1]]])


    forecast_b = np.array([[[0,0.5],[0.5,1]], 
                        [[0.5,0],[1,0.5]]])


    groundtruth = forecast_a 

    # example 3
    forecast_a = np.array([[[0,0],[0.5,0]], 
                        [[0.5,1],[1,1]]])


    forecast_b = np.array([[[0,1],[0.5,1]], 
                        [[0.5,0],[1,0]]])


    groundtruth = forecast_a 

    # other
    # forecast_a = np.array([[[0,0],[1,0]], 
    #                        [[0,1],[1,1]]],
    #                        dtype='float')

    # forecast_b = np.array([[[0,0],[1,0]], 
    #                        [[0,1],[1,1]]],
    #                        dtype='float')
    # forecast_b[:,:,1]+=0.1
    # groundtruth = forecast_a 

## plotting
fig, axes = plt.subplots(1,3, figsize=(15,5), sharey=True, )
plt.subplots_adjust(wspace=0.1)
axes[0].set_title('Forecast A')
axes[0].plot(forecast_a[:,:,0].T, forecast_a[:,:,1].T, 'x-')
for s in range(2):
    for t in range(2):
        if t == 0:
            offset=0.05
        else:
            offset=-0.13

        if toyexample==1:
            offset /= 2

        axes[0].text(forecast_a[s,t,0]+offset, forecast_a[s,t,1]+0.03, f"t={t+1}")
axes[0].grid(alpha=0.25)
axes[0].set_aspect('equal', 'box')
axes[0].set_xlabel('$s=1$')
axes[0].set_ylabel('$s=2$')
axes[1].set_title('Forecast B')
axes[1].plot(forecast_b[:,:,0].T, forecast_b[:,:,1].T, 'x-')
for s in range(2):
    for t in range(2):
        if t == 0:
            offset=0.05
        else:
            offset=-0.13

        if toyexample==1:
            offset /= 2
       
        axes[1].text(forecast_b[s,t,0]+offset, forecast_b[s,t,1]+0.03, f"t={t+1}")
axes[1].grid(alpha=0.25) 
axes[1].set_aspect('equal', 'box')
axes[1].set_xlabel('$s=1$')
# axes[1].set_ylabel='$s=2$')
# plt.savefig(f"figs/es_variations/toy_example{toyexample}.pdf", bbox_inches="tight", )
# plt.show()


# conform to the dimensionality for my energy score implementation 
forecast_a = forecast_a.reshape(1, 2, 2, 2).astype('float')
forecast_b = forecast_b.reshape(1, 2, 2, 2).astype('float')
observations = groundtruth[obs_i]
observations = observations.reshape(len(obs_i),1,2,2).astype('float')
# constant prediction over observations
forecasts_a = np.repeat(forecast_a, len(obs_i), axis=0)
forecasts_b = np.repeat(forecast_b, len(obs_i), axis=0)
print(forecasts_a.shape)

est_as = []
est_bs = []
ess_as = []
ess_bs = []
es_as = []
es_bs = []

for i in range(len(obs_i)):
    y = observations[None,i]
    x_a = forecasts_a[None,i]
    x_b = forecasts_b[None,i]
    d = np.product(y.shape[2:])
    #
    est_a = energy_score_temporal(y, x_a, K=x_a.shape[1], d=y.shape[-2])
    est_b = energy_score_temporal(y, x_b, K=x_b.shape[1], d=y.shape[-2])
    ess_a = energy_score_spatial(y, x_a, K=x_a.shape[1], d=y.shape[-1])
    ess_b = energy_score_spatial(y, x_b, K=x_b.shape[1], d=y.shape[-1])
    es_a = energy_score(y, x_a, K=x_a.shape[1], d=d)
    es_b = energy_score(y, x_b, K=x_b.shape[1], d=d)
    #
    est_as.append(est_a.ravel())
    est_bs.append(est_b.ravel())
    ess_as.append(ess_a.ravel())
    ess_bs.append(ess_b.ravel())
    es_as.append(es_a.ravel())
    es_bs.append(es_b.ravel())

# plt.figure(figsize=(5,5))
# plt.bar(0, np.mean(est_as))
# plt.bar(1, np.mean(est_bs))
# plt.bar(2, np.mean(ess_as))
# plt.bar(3, np.mean(ess_bs))
# plt.bar(4, np.mean(es_as))
# plt.bar(5, np.mean(es_bs))
# plt.xticks([0, 1, 2, 3, 4, 5], ['EST(A)', 'EST(B)', 'ESS(A)', 'ESS(A)', 'ES(A)', 'ES(B)'], rotation=0, fontsize=12)
# plt.grid(alpha=0.25 )
# plt.xlabel(" ")
# plt.ylabel(" ")
# plt.savefig(f"figs/es_variations/toy_example{toyexample}_scores.pdf", bbox_inches="tight", )
# plt.show()

axes[2].set_title("Energy Scores")
axes[2].bar(0, np.mean(est_as))
axes[2].bar(1, np.mean(est_bs))
axes[2].bar(2, np.mean(ess_as))
axes[2].bar(3, np.mean(ess_bs))
axes[2].bar(4, np.mean(es_as))
axes[2].bar(5, np.mean(es_bs))
axes[2].text(0-0.45, np.mean(est_as)+0.01, f"{np.mean(est_as):2.3f}", rotation=0)
axes[2].text(1-0.45, np.mean(est_bs)+0.01, f"{np.mean(est_bs):2.3f}", rotation=0)
axes[2].text(2-0.45, np.mean(ess_as)+0.01, f"{np.mean(ess_as):2.3f}", rotation=0)
axes[2].text(3-0.45, np.mean(ess_bs)+0.01, f"{np.mean(ess_bs):2.3f}", rotation=0)
axes[2].text(4-0.45, np.mean(es_as) +0.01, f"{np.mean(es_as):2.3f}" , rotation=0)
axes[2].text(5-0.45, np.mean(es_bs) +0.01, f"{np.mean(es_bs):2.3f}" , rotation=0)
axes[2].set_xticks([0, 1, 2, 3, 4, 5], ['EST(A)', 'EST(B)', 'ESS(A)', 'ESS(A)', 'ES(A)', 'ES(B)'], rotation=45, fontsize=12)
axes[2].grid(alpha=0.25 )
axes[2].set_xlabel(" ")
axes[2].set_ylabel(" ")
axes[2].set_ylim([-0.05, 1.1])
# axes[2].set_aspect('equal', 'box')
plt.savefig(f"figs/es_variations/toy_example{toyexample}_all.pdf", bbox_inches="tight",)
plt.show()

print(f"EST a: {np.mean(est_as):2.3f}")
print(f"EST b: {np.mean(est_bs):2.3f}")
print(f"ESS a: {np.mean(ess_as):2.3f}")
print(f"ESS b: {np.mean(ess_bs):2.3f}")
print(f"ES  a: {np.mean(es_as):2.3f}")
print(f"ES  b: {np.mean(es_bs):2.3f}")


#%%                        
# from scipy.optimize import minimize

# def opt_func(forecast_b):
#     est_as = []
#     est_bs = []
#     ess_as = []
#     ess_bs = []
#     #
#     for obs in observations:
#         # conform to the dimensionality for my energy score implementation 
#         # n_sample,s n_scenarios, n_horizon, n_dims
#         y = obs.reshape(1, 1, 2, 1).astype('float')
#         # forecasts are constant for new obs
#         x_a = forecast_a.reshape(1, 2, 2, 1).astype('float') 
#         x_b = forecast_b.reshape(1, 2, 2, 1).astype('float')
#         d = np.product(y.shape[2:])
#         #
#         est_a = energy_score_temporal(y, x_a, K=x_a.shape[1], d=d)
#         est_b = energy_score_temporal(y, x_b, K=x_b.shape[1], d=d)
#         ess_a = energy_score_spatial(y, x_a, K=x_a.shape[1], d=d)
#         ess_b = energy_score_spatial(y, x_b, K=x_b.shape[1], d=d)
#         #
#         est_as.append(est_a.ravel())
#         est_bs.append(est_b.ravel())
#         ess_as.append(ess_a.ravel())
#         ess_bs.append(ess_b.ravel())
#     # print(f"EST a: {np.mean(est_as):2.3f}")
#     # print(f"EST b: {np.mean(est_bs):2.3f}")
#     # print(f"ESS a: {np.mean(ess_as):2.3f}")
#     # print(f"ESS b: {np.mean(ess_bs):2.3f}")
#     return np.abs(np.mean(est_as)-np.mean(est_bs))+1/(np.abs(np.mean(ess_as)-np.mean(ess_bs))+1e-5)
#     # return np.abs(np.mean(ess_as)-np.mean(ess_bs)) # 1/(np.abs(np.mean(est_as)-np.mean(est_bs))+1e-7)

# res = minimize(opt_func, 
#                x0=forecast_b.flatten(), 
#                bounds=[(0,1.1),(0,1.1),(0,1.1),(0,1.1)], 
#                options={'gtol': 1e-6, 'disp': True})
# res.x

#%%