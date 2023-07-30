#%%
import sympy as sp
from sympy.plotting import plot 

def find_solutions(x1_g, y1_g, x2_g, y2_g, c):
    x1, y1, x2, y2 = sp.symbols('x1 y1 x2 y2')
    equation = (x1 - x1_g)**2 + (y1 - y1_g)**2 + (x2 - x2_g)**2 + (y2 - y2_g)**2 - c
    solutions = sp.solve(equation, (x1, y1, x2, y2))
    return solutions

# Example usage
x1_g = 2
y1_g = 3
x2_g = 5
y2_g = 7
c = 50

solutions = find_solutions(x1_g, y1_g, x2_g, y2_g, c)

print("Solutions:")
for solution in solutions:
    print(solution)

# solutions[0].evalf()

#%%
import numpy as np
from matplotlib import pyplot as plt

def calculate_displacement_error(y, y_hat):
    # y_hat dim: (num_samples x num_scenarios x num_leadtime x 2)
    return np.sqrt(np.sum((y-y_hat)**2, axis=-1)) 

def generate_trajectory(p0=np.array([0,0]), 
                        v0=np.array([0.75,0.75]), 
                        a0=np.array([0.1,0.1]), 
                        seed=None):
    if seed is not None:
        np.random.seed(seed)

    positions = []
    velocities = []
    accelerations = []

    positions.append(p0)
    velocities.append(v0)
    accelerations.append(a0)

    frames = 12 # 
    dt = 1.0 # delta time of each frame in seconds
    for f in range(1,frames):
        t = f*dt
        a_t = accelerations[-1] + np.random.normal(0, 0.01, size=2)
        np.clip(a_t, -0.7, 0.7)
        v_t = v0 + a_t*t #+ np.random.normal(0, 0.05, size=2)
        np.clip(v_t, -2, 2)
        p_t = p0 + v_t*t + 0.5*a_t*t**2
        positions.append(p_t)
        velocities.append(v_t)
        accelerations.append(a_t)

    positions = np.array(positions)
    # print(positions)
    return positions

p0=np.array([0,0])
v0=np.array([0.75,0.75])
a0=np.array([0.1,0.1])

ground_truth = generate_trajectory(p0=p0, v0=v0, a0=a0, seed=0,)

trajectories = []
for t in range(100):
    traj = generate_trajectory(p0=p0, v0=v0, a0=a0,)
    trajectories.append(traj)

trajectories = np.array(trajectories)
print(trajectories.shape)
#%%

similar_idx = np.where(calculate_displacement_error(ground_truth, trajectories).mean(axis=-1) < 0.1)
trajectories_similar = trajectories[similar_idx]

#%%
plt.plot(ground_truth[:,0], ground_truth[:,1], 'o-')
# plt.plot(trajectories[:,:,0], trajectories[:,:,1], 'o-', color='red', alpha=0.1)
plt.plot(trajectories_similar[:,:,0], trajectories_similar[:,:,1], 'o-', color='red', alpha=0.1)
plt.show()