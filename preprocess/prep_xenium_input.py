import numpy as np
import matplotlib.pyplot as pl
import scipy.stats as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings("ignore")

#calculate KDE
patch_dir = 'data/patch'

rna_dir = os.path.join(patch_dir, 'rna')
save_dir = os.path.join(patch_dir, 'kde')

os.makedirs(save_dir, exist_ok=True)
rna_list = os.listdir(rna_dir)
rna_list = [rna.split('.')[0] for rna in rna_list]

def process_patch(rna):
    rna_patch = pd.read_csv(os.path.join(rna_dir, rna+'.csv'))
    rna_patch['y_pixel'] = 512 - rna_patch['y_pixel']
    x = rna_patch['x_pixel']
    y = rna_patch['y_pixel']
    xmin, xmax = 0, 512
    ymin, ymax = 0, 512
    X, Y = np.mgrid[xmin:xmax:512j, ymin:ymax:512j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    #plot
    fig, ax = pl.subplots()
    ax.imshow(np.rot90(Z), cmap=pl.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    ax.plot(x, y, 'k,', markersize=0.5)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, '{}.png'.format(rna)), bbox_inches='tight', pad_inches=0,dpi=138.7)

    # print(rna)

num_workers = 30 # or os.cpu_count() to use all available cores
with ProcessPoolExecutor(max_workers=num_workers) as executor:
    results = list(executor.map(process_patch, rna_list))
# process_patch(rna_list[0])
print("Processing complete!")

for rna in tqdm(rna_list):
    rna_patch = pd.read_csv(os.path.join(rna_dir, rna+'.csv'))
    rna_patch['y_pixel'] = 512 - rna_patch['y_pixel']
    x = rna_patch['x_pixel']
    y = rna_patch['y_pixel']
    xmin, xmax = 0, 512
    ymin, ymax = 0, 512
    X, Y = np.mgrid[xmin:xmax:512j, ymin:ymax:512j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    Z = np.reshape(kernel(positions).T, X.shape)

    #plot
    fig, ax = pl.subplots()
    ax.imshow(np.rot90(Z), cmap=pl.cm.gist_earth_r, extent=[xmin, xmax, ymin, ymax])
    ax.plot(x, y, 'k.', markersize=1)
    plt.axis('off')
    plt.savefig(os.path.join(save_dir, '{}.png'.format(rna)), bbox_inches='tight', pad_inches=0,dpi=138.7)
