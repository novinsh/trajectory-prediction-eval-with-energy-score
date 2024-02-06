#%%
import numpy as np
import pickle
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
from string import ascii_lowercase

filenames = ['npsn-sgcn-mc.pkl',
             'npsn-sgcn-qmc.pkl',
             'npsn-sgcn-npsn.pkl',
             'npsn-stgcnn-mc.pkl',
             'npsn-stgcnn-qmc.pkl',
             'npsn-stgcnn-npsn.pkl',
             'npsn-pecnet-mc.pkl', 
             'npsn-pecnet-qmc.pkl',
             'npsn-pecnet-npsn.pkl', 
            ]

results = {filename.split('.')[0] : None for filename in filenames}
for filename in filenames:
    with open(filename, 'rb') as f:
        arr = np.array(pickle.load(f))
        # append mean 
        arr = np.concatenate((arr, arr.mean(axis=1).reshape(-1,1)), axis=1) # mean
        results[filename.split('.')[0]] = arr

#%%
results_df = {filename.split('.')[0] : None for filename in filenames}
for name, result in results.items():
    df = pd.DataFrame(result.T, columns=['ADE', 'FDE', 'TCC', 'ES'], index=['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2', 'AVG'])
    df.name = name
    results_df[name] = df
    print(name)
    df.round(3)
    display(df)

dataset_names = ['ETH', 'HOTEL', 'UNIV', 'ZARA1', 'ZARA2', 'AVG']
results_all_models = pd.DataFrame([], columns=dataset_names)
for name, result in results.items():
    # result rows: 'ADE', 'FDE', 'TCC', 'ES' and columns are datasets
    results_all_models.loc[name] = [f'{result[0,i]:2.2f}/{result[1,i]:2.2f}/{result[3,i]:2.2f}' for i in range(len(dataset_names))]
results_all_models


#%%

def plot_boxplot_avg_result(title=''):
    results_all = pd.DataFrame([], columns=dataset_names)
    for name, result in results.items():
        # result rows: 'ADE', 'FDE', 'TCC', 'ES' and columns are datasets
        if title=='ADE':
            results_all.loc[name] = [f'{result[0,i]:2.2f}' for i in range(len(dataset_names))]
        elif title=='FDE':
            results_all.loc[name] = [f'{result[1,i]:2.2f}' for i in range(len(dataset_names))]
        elif title=='ES':
            results_all.loc[name] = [f'{result[3,i]:2.2f}' for i in range(len(dataset_names))]
        else:
            assert False
    # results_all

    _, ax = plt.subplots(1, 1)
    ax.set_title(title)
    boxplot = ax.boxplot(results_all.loc[:,dataset_names[:-1]].T.astype('float32'), 
                vert=True,
            patch_artist=True,
            labels=[f'({c})' for c in ascii_lowercase[:len(filenames)]])
    # results_all.loc[:,dataset_names[:-1]].T.astype('float32').boxplot()
    # sns.boxplot(data=results_all.loc[:,dataset_names[:-1]].T.astype('float32'), x='')
    # plt.xticks(range(1,len(filenames)+1), [f'({c})' for c in ascii_lowercase[:len(filenames)]])

    colors=['r', 'b', 'g', 'r', 'b', 'g', 'r', 'b', 'g']
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
    ax.yaxis.grid(True)
    plt.show()

plot_boxplot_avg_result('ADE')
plot_boxplot_avg_result('FDE')
plot_boxplot_avg_result('ES')
#%%

results_all_ade = pd.DataFrame([], columns=dataset_names)
for name, result in results.items():
    # result rows: 'ADE', 'FDE', 'TCC', 'ES' and columns are datasets
    results_all_ade.loc[name] = [f'{result[1,i]:2.2f}' for i in range(len(dataset_names))]
results_all_ade


plt.title('FDE')
results_all_ade.loc[:,dataset_names[:-1]].T.astype('float32').boxplot()
# sns.boxplot(data=results_all_ade.loc[:,dataset_names[:-1]].T.astype('float32'), x='')
plt.xticks(range(1,len(filenames)+1), [f'({c})' for c in ascii_lowercase[:len(filenames)]])
plt.show()


results_all_ade = pd.DataFrame([], columns=dataset_names)
for name, result in results.items():
    # result rows: 'ADE', 'FDE', 'TCC', 'ES' and columns are datasets
    results_all_ade.loc[name] = [f'{result[3,i]:2.2f}' for i in range(len(dataset_names))]
results_all_ade


plt.title('ES')
results_all_ade.loc[:,dataset_names[:-1]].T.astype('float32').boxplot()
# sns.boxplot(data=results_all_ade.loc[:,dataset_names[:-1]].T.astype('float32'), x='')
plt.xticks(range(1,len(filenames)+1), [f'({c})' for c in ascii_lowercase[:len(filenames)]])
plt.show()



#%%
results_all_models.query('not index.str.contains("qmc")').sort_index(key=lambda x: [xx.split("-")[-1] for xx in x])
print(
    results_all_models \
      .query('not index.str.contains("qmc")').sort_index(key=lambda x: [xx.split("-")[-1] for xx in x]) \
      .to_latex(
                                    index=True,
                                    formatters={"name": str.upper},
                                    float_format="{:.1f}".format,
                                    )
)

print(
    results_all_models \
      .sort_index(key=lambda x: [xx.split("-")[-1] for xx in x]) \
      .to_latex(
                                    index=True,
                                    formatters={"name": str.upper},
                                    float_format="{:.1f}".format,
                                    )
)


#%%
def gain(ref, fcst):
    return ((ref-fcst) / ref)*100

print(gain(results_df['npsn-sgcn-mc']['ADE'], results_df['npsn-sgcn-qmc']['ADE']))
print(gain(results_df['npsn-sgcn-mc']['ADE'], results_df['npsn-sgcn-npsn']['ADE']))

#%%

print(gain(results_df['npsn-sgcn-mc']['ES'], results_df['npsn-sgcn-qmc']['ES']))
print(gain(results_df['npsn-sgcn-mc']['ES'], results_df['npsn-sgcn-npsn']['ES']))

#%%


# gain(results_df['npsn-sgcn-mc']['ES'], results_df['npsn-sgcn-npsn']['ES'])

print(results_df['npsn-sgcn-mc'].loc['AVG','ES'])
print(results_df['npsn-sgcn-qmc'].loc['AVG','ES'])
print(results_df['npsn-sgcn-npsn'].loc['AVG','ES'])

print()

print(results_df['npsn-stgcnn-mc'].loc['AVG','ES'])
print(results_df['npsn-stgcnn-qmc'].loc['AVG','ES'])
print(results_df['npsn-stgcnn-npsn'].loc['AVG','ES'])

print()

print(results_df['npsn-pecnet-mc'].loc['AVG','ES'])
print(results_df['npsn-pecnet-qmc'].loc['AVG','ES'])
print(results_df['npsn-pecnet-npsn'].loc['AVG','ES'])

#%%
print(results_df['npsn-sgcn-npsn'].loc['AVG','ES'])
print(results_df['npsn-stgcnn-npsn'].loc['AVG','ES'])
print(results_df['npsn-pecnet-npsn'].loc['AVG','ES'])
