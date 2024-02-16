import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt
import matplotlib as mpl

n = 10000
mpl.rcParams['axes.titlesize'] = 18  # Title size
mpl.rcParams['axes.labelsize'] = 16  # Label size
mpl.rcParams['xtick.labelsize'] = 14  # X-axis tick size
mpl.rcParams['ytick.labelsize'] = 14  # Y-axis tick size
mpl.rcParams['font.size'] = 13       # Global font size

# Set line width for plot curves
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['lines.markersize'] = 8  # Marker size
mpl.rcParams['lines.markeredgewidth'] = 2  # Marker width

values = np.random.default_rng().normal(0, 1, n).reshape((n,1))
x1 = np.random.default_rng().normal(0, 1, 3*n).reshape((1, 3*n))
x2 = np.random.default_rng().normal(0, 10, 3*n).reshape((1, 3*n))
x3 = np.random.default_rng().normal(0, 0.1, 3*n).reshape((1, 3*n))
x4 = np.random.default_rng().normal(1, 1, 3*n).reshape((1, 3*n))
x5 = np.random.default_rng().uniform(-2, 2, 3*n).reshape((1, 3*n))
x6 = np.random.default_rng().normal(0, 1.75, 3*n).reshape((1, 3*n))
diff1 = np.abs(values - x1)
diff2 = np.abs(values - x2)
diff3 = np.abs(values - x3)
diff4 = np.abs(values - x4)
diff5 = np.abs(values - x5)
diff6 = np.abs(values - x6)
minimum_values1 = []
minimum_values2 = []
minimum_values3 = []
minimum_values4 = []
minimum_values5 = []
minimum_values6 = []
for i in trange(0, 3*n+1, 100):
    if i == 0:
        mins1 = np.min(diff1[:,:1], axis = 1)
        mins2 = np.min(diff2[:,:1], axis = 1)
        mins3 = np.min(diff3[:,:1], axis = 1)
        mins4 = np.min(diff4[:,:1], axis = 1)
        mins5 = np.min(diff5[:,:1], axis = 1)
        mins6 = np.min(diff6[:,:1], axis = 1)
    else:
        mins1 = np.min(diff1[:,:i], axis = 1)
        mins2 = np.min(diff2[:,:i], axis = 1)
        mins3 = np.min(diff3[:,:i], axis = 1)
        mins4 = np.min(diff4[:,:i], axis = 1)
        mins5 = np.min(diff5[:,:i], axis = 1)
        mins6 = np.min(diff6[:,:i], axis = 1)
           
    minimum_values1.append(np.mean(mins1))
    minimum_values2.append(np.mean(mins2))
    minimum_values3.append(np.mean(mins3))
    minimum_values4.append(np.mean(mins4))
    minimum_values5.append(np.mean(mins5))
    minimum_values6.append(np.mean(mins6))
values = np.random.default_rng().normal(0, 300, n).reshape((n,1))
x1 = np.random.default_rng().normal(0, 300, 3*n).reshape((1, 3*n))
x2 = np.random.default_rng().normal(0, 3000, 3*n).reshape((1, 3*n))
x3 = np.random.default_rng().normal(0, 30, 3*n).reshape((1, 3*n))
x4 = np.random.default_rng().normal(10, 300, 3*n).reshape((1, 3*n))
x5 = np.random.default_rng().uniform(-600, 600, 3*n).reshape((1, 3*n))
x6 = np.random.default_rng().normal(0, 525, 3*n).reshape((1, 3*n))
diff1 = np.abs(values - x1)
diff2 = np.abs(values - x2)
diff3 = np.abs(values - x3)
diff4 = np.abs(values - x4)
diff5 = np.abs(values - x5)
diff6 = np.abs(values - x6)
minimum_values11 = []
minimum_values12 = []
minimum_values13 = []
minimum_values14 = []
minimum_values15 = []
minimum_values16 = []
for i in trange(0, 3*n+1, 100):
    if i == 0:
        mins1 = np.min(diff1[:,:1], axis = 1)
        mins2 = np.min(diff2[:,:1], axis = 1)
        mins3 = np.min(diff3[:,:1], axis = 1)
        mins4 = np.min(diff4[:,:1], axis = 1)
        mins5 = np.min(diff5[:,:1], axis = 1)
        mins6 = np.min(diff6[:,:1], axis = 1)
    else:
        mins1 = np.min(diff1[:,:i], axis = 1)
        mins2 = np.min(diff2[:,:i], axis = 1)
        mins3 = np.min(diff3[:,:i], axis = 1)
        mins4 = np.min(diff4[:,:i], axis = 1)
        mins5 = np.min(diff5[:,:i], axis = 1)
        mins6 = np.min(diff6[:,:i], axis = 1)
           
    minimum_values11.append(np.mean(mins1))
    minimum_values12.append(np.mean(mins2))
    minimum_values13.append(np.mean(mins3))
    minimum_values14.append(np.mean(mins4))
    minimum_values15.append(np.mean(mins5))
    minimum_values16.append(np.mean(mins6))
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].plot([i for i in range(0, 3*n+1, 100)], minimum_values1, color="blue", label="N(0,1)")
axes[0].plot([i for i in range(0, 3*n+1, 100)], minimum_values6, color="violet", label="N(0,1.75)")
axes[0].plot([i for i in range(0, 3*n+1, 100)], minimum_values2, color="red", label="N(0,10)")
axes[0].plot([i for i in range(0, 3*n+1, 100)], minimum_values3, color="brown", label="N(0,0.1)")
axes[0].plot([i for i in range(0, 3*n+1, 100)], minimum_values4, color="orange", label="N(1,1)")
axes[0].plot([i for i in range(0, 3*n+1, 100)], minimum_values5, color="green", label="U(-2,2)")
axes[0].set_ylim(0,0.55)
axes[0].set_xlim(0,30000)
axes[0].set_xlabel("K")
axes[0].set_ylabel("minFDE")
axes[0].set_xticks(range(0,30001, 4000))
axes[0].grid(alpha=0.25)
axes[0].legend()
axes[0].set_title("True distribution N(0,1)")

axes[1].plot([i for i in range(0, 3*n+1, 100)], minimum_values1, color="blue", label="N(0,1)")
axes[1].plot([i for i in range(0, 3*n+1, 100)], minimum_values6, color="violet", label="N(0,1.75)")
axes[1].plot([i for i in range(0, 3*n+1, 100)], minimum_values2, color="red", label="N(0,10)")
axes[1].plot([i for i in range(0, 3*n+1, 100)], minimum_values4, color="orange", label="N(1,1)")
axes[1].set_ylim(0,0.005)
axes[1].set_xlim(0,30000)
axes[1].set_xlabel("K")
axes[1].set_xticks(range(0,30001, 4000))
axes[1].grid(alpha=0.25)
axes[1].legend()
axes[1].set_title("True distribution N(0,1)")

axes[2].plot([i for i in range(0, 3*n+1, 100)], minimum_values11, color="blue", label="N(0,300)")
axes[2].plot([i for i in range(0, 3*n+1, 100)], minimum_values16, color="violet", label="N(0,525)")
axes[2].plot([i for i in range(0, 3*n+1, 100)], minimum_values12, color="red", label="N(0,3000)")
axes[2].plot([i for i in range(0, 3*n+1, 100)], minimum_values13, color="brown", label="N(0,30)")
axes[2].plot([i for i in range(0, 3*n+1, 100)], minimum_values14, color="orange", label="N(10,300)")
axes[2].plot([i for i in range(0, 3*n+1, 100)], minimum_values15, color="green", label="U(-600,600)")
axes[2].set_ylim(0,170)
axes[2].set_xlim(0,30000)
axes[2].set_xlabel("K")
axes[2].set_xticks(range(0,30001, 4000))
axes[2].grid(alpha=0.25)
axes[2].legend()
axes[2].set_title("True distribution N(0,300)")

axes[3].plot([i for i in range(0, 3*n+1, 100)], minimum_values11, color="blue", label="N(0,300)")
axes[3].plot([i for i in range(0, 3*n+1, 100)], minimum_values16, color="violet", label="N(0,525)")
axes[3].plot([i for i in range(0, 3*n+1, 100)], minimum_values12, color="red", label="N(0,3000)")
axes[3].plot([i for i in range(0, 3*n+1, 100)], minimum_values14, color="orange", label="N(10,300)")
axes[3].set_ylim(0,0.55)
axes[3].set_xlim(0,30000)
axes[3].set_xlabel("K")
axes[3].set_xticks(range(0,30001, 4000))
axes[3].grid(alpha=0.25)
axes[3].legend()
axes[3].set_title("True distribution N(0,300)")

plt.tight_layout()

plt.savefig("minFDE_differentK.pdf")

#############################
#############################
#############################

values = np.zeros((6,300))
y = np.random.default_rng().normal(0, 1, 30000)
for k, var in enumerate([1, 1.75, 10, 0.1]):
    x1 = np.random.default_rng().normal(0, var, 30000)
    x2 = np.random.default_rng().normal(0, var, 30000)
    diffs2 = np.abs(x1 - y)

    x1 = x1.reshape(30000,1)
    x2 = x2.reshape(1,30000)

    diffs = np.abs(x1 - x2)

    for i in range(1, 301):
        diff1 = np.mean(diffs[:(i*100), :(i*100)])
        diff2 = np.mean(diffs2[:(i*100)])
        values[k, i-1] = diff2 - 0.5 * diff1
x1 = np.random.default_rng().normal(1, 1, 30000)
x2 = np.random.default_rng().normal(1, 1, 30000)
diffs2 = np.abs(x1 - y)

x1 = x1.reshape(30000,1)
x2 = x2.reshape(1,30000)

diffs = np.abs(x1 - x2)

for i in range(1, 301):
    diff1 = np.mean(diffs[:(i*100), :(i*100)])
    diff2 = np.mean(diffs2[:(i*100)])
    values[4, i-1] = diff2 - 0.5 * diff1

x1 = np.random.default_rng().uniform(-2, 2, 30000)
x2 = np.random.default_rng().uniform(-2, 2, 30000)
diffs2 = np.abs(x1 - y)

x1 = x1.reshape(30000,1)
x2 = x2.reshape(1,30000)

diffs = np.abs(x1 - x2)

for i in range(1, 301):
    diff1 = np.mean(diffs[:(i*100), :(i*100)])
    diff2 = np.mean(diffs2[:(i*100)])
    values[5, i-1] = diff2 - 0.5 * diff1
values_2 = np.zeros((6,300))
y = np.random.default_rng().normal(0, 300, 30000)
for k, var in enumerate([300, 525, 3000, 30]):
    x1 = np.random.default_rng().normal(0, var, 30000)
    x2 = np.random.default_rng().normal(0, var, 30000)
    diffs2 = np.abs(x1 - y)

    x1 = x1.reshape(30000,1)
    x2 = x2.reshape(1,30000)

    diffs = np.abs(x1 - x2)

    for i in range(1, 301):
        diff1 = np.mean(diffs[:(i*100), :(i*100)])
        diff2 = np.mean(diffs2[:(i*100)])
        values_2[k, i-1] = diff2 - 0.5 * diff1

x1 = np.random.default_rng().normal(10, 300, 30000)
x2 = np.random.default_rng().normal(10, 300, 30000)
diffs2 = np.abs(x1 - y)

x1 = x1.reshape(30000,1)
x2 = x2.reshape(1,30000)

diffs = np.abs(x1 - x2)

for i in range(1, 301):
    diff1 = np.mean(diffs[:(i*100), :(i*100)])
    diff2 = np.mean(diffs2[:(i*100)])
    values_2[4, i-1] = diff2 - 0.5 * diff1

x1 = np.random.default_rng().uniform(-600, 600, 30000)
x2 = np.random.default_rng().uniform(-600, 600, 30000)
diffs2 = np.abs(x1 - y)

x1 = x1.reshape(30000,1)
x2 = x2.reshape(1,30000)

diffs = np.abs(x1 - x2)

for i in range(1, 301):
    diff1 = np.mean(diffs[:(i*100), :(i*100)])
    diff2 = np.mean(diffs2[:(i*100)])
    values_2[5, i-1] = diff2 - 0.5 * diff1
fig, axes = plt.subplots(1, 4, figsize=(24, 6))

axes[0].plot([i for i in range(100, 3*n+1, 100)], values[0,:], color="blue", label="N(0,1)")
axes[0].plot([i for i in range(100, 3*n+1, 100)], values[1,:], color="violet", label="N(0,1.75)")
axes[0].plot([i for i in range(100, 3*n+1, 100)], values[2,:], color="red", label="N(0,10)")
axes[0].plot([i for i in range(100, 3*n+1, 100)], values[3,:], color="brown", label="N(0,0.1)")
axes[0].plot([i for i in range(100, 3*n+1, 100)], values[4,:], color="orange", label="N(1,1)")
axes[0].plot([i for i in range(100, 3*n+1, 100)], values[5,:], color="green", label="U(-2,2)")
axes[0].set_ylim(0,4)
axes[0].set_xlim(0,30000)
axes[0].set_xlabel("K")
axes[0].set_ylabel("ES")
axes[0].set_xticks(range(0,30001, 4000))
axes[0].grid(alpha=0.25)
axes[0].legend()
axes[0].set_title("True distribution N(0,1)")

axes[1].plot([i for i in range(100, 3*n+1, 100)], values[0,:], color="blue", label="N(0,1)")
axes[1].plot([i for i in range(100, 3*n+1, 100)], values[1,:], color="violet", label="N(0,1.75)")
axes[1].plot([i for i in range(100, 3*n+1, 100)], values[2,:], color="red", label="N(0,10)")
axes[1].plot([i for i in range(100, 3*n+1, 100)], values[3,:], color="brown", label="N(0,0.1)")
axes[1].plot([i for i in range(100, 3*n+1, 100)], values[4,:], color="orange", label="N(1,1)")
axes[1].plot([i for i in range(100, 3*n+1, 100)], values[5,:], color="green", label="U(-2,2)")
axes[1].set_ylim(0.45,1)
axes[1].set_xlim(0,30000)
axes[1].set_xlabel("K")
axes[1].set_xticks(range(0,30001, 4000))
axes[1].grid(alpha=0.25)
axes[1].legend()
axes[1].set_title("True distribution N(0,1)")

axes[2].plot([i for i in range(100, 3*n+1, 100)], values_2[0,:], color="blue", label="N(0,300)")
axes[2].plot([i for i in range(100, 3*n+1, 100)], values_2[1,:], color="violet", label="N(0,525)")
axes[2].plot([i for i in range(100, 3*n+1, 100)], values_2[2,:], color="red", label="N(0,3000)")
axes[2].plot([i for i in range(100, 3*n+1, 100)], values_2[3,:], color="brown", label="N(0,30)")
axes[2].plot([i for i in range(100, 3*n+1, 100)], values_2[4,:], color="orange", label="N(10,300)")
axes[2].plot([i for i in range(100, 3*n+1, 100)], values_2[5,:], color="green", label="U(-600,600)")
axes[2].set_ylim(150,750)
axes[2].set_xlim(0,30000)
axes[2].set_xlabel("K")
axes[2].set_xticks(range(0,30001, 4000))
axes[2].grid(alpha=0.25)
axes[2].legend()
axes[2].set_title("True distribution N(0,300)")

axes[3].plot([i for i in range(100, 3*n+1, 100)], values_2[0,:], color="blue", label="N(0,300)")
axes[3].plot([i for i in range(100, 3*n+1, 100)], values_2[1,:], color="violet", label="N(0,525)")
axes[3].plot([i for i in range(100, 3*n+1, 100)], values_2[2,:], color="red", label="N(0,3000)")
axes[3].plot([i for i in range(100, 3*n+1, 100)], values_2[3,:], color="brown", label="N(0,30)")
axes[3].plot([i for i in range(100, 3*n+1, 100)], values_2[4,:], color="orange", label="N(10,300)")
axes[3].plot([i for i in range(100, 3*n+1, 100)], values_2[5,:], color="green", label="U(-600,600)")
axes[3].set_ylim(165,270)
axes[3].set_xlim(0,30000)
axes[3].set_xlabel("K")
axes[3].set_xticks(range(0,30001, 4000))
axes[3].grid(alpha=0.25)
axes[3].legend()
axes[3].set_title("True distribution N(0,300)")

plt.tight_layout()

plt.savefig("ES_differentK.pdf")