import pandas as pd
import numpy as np
import cv2
from skimage import io
from PIL import Image 
import os
from tqdm import tqdm
from pathlib import Path
import shutil
import matplotlib.pyplot as plt
import os
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
import json
import warnings
warnings.filterwarnings("ignore")

Image.MAX_IMAGE_PIXELS = 1000000000 

img_size = 512

fov_dir = 'data/xenium_lung/'

mask_dir = os.path.join(fov_dir, 'cell_mask')
mask_10x_dir = os.path.join(fov_dir, 'mask_10x')
dapi_dir = os.path.join(fov_dir, 'img_dapi')
trans_dir = os.path.join(fov_dir, 'transcript')
seg_img_dir = os.path.join(fov_dir, 'img_for_seg')

output_dir = 'data/patch'

os.makedirs(output_dir, exist_ok=True)
output_mask_dir = os.path.join(output_dir, 'cell_mask')
output_dapi_dir = os.path.join(output_dir, 'dapi')
output_rna_dir = os.path.join(output_dir, 'rna')
rna_vis_dir = os.path.join(output_dir, 'rna_vis')
img_seg_dir = os.path.join(output_dir, 'img_seg')
os.makedirs(output_mask_dir, exist_ok=True)
os.makedirs(output_dapi_dir, exist_ok=True)
os.makedirs(output_rna_dir, exist_ok=True)
os.makedirs(rna_vis_dir, exist_ok=True)
os.makedirs(img_seg_dir, exist_ok=True)

roi_list = os.listdir(mask_dir)
roi_list = [roi.split('.')[0] for roi in roi_list]

id = 0
n_cells = 0
for roi in tqdm(roi_list):
    # roi='C2'
    mask = io.imread(os.path.join(mask_dir, roi+'.tif'))
    mask_10x = io.imread(os.path.join(mask_10x_dir, roi+'.tif'))
    dapi = io.imread(os.path.join(dapi_dir, roi+'.png'))
    rna = pd.read_csv(os.path.join(trans_dir, roi+'.csv'))
    seg_img = io.imread(os.path.join(seg_img_dir, roi+'.png'))

    w = mask.shape[0]//img_size
    h = mask.shape[1]//img_size

    for i in range(w):
        for j in range(h):
            mask_patch = mask[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size]
            mask_10x_patch = mask_10x[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size]
            dapi_patch = dapi[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size]
            rna_patch = rna[(rna['x_pixel'] >= j*img_size) & (rna['x_pixel'] < (j+1)*img_size) & (rna['y_pixel'] >= i*img_size) & (rna['y_pixel'] < (i+1)*img_size)]
            rna_patch['x_pixel'] = rna_patch['x_pixel'] - j*img_size
            rna_patch['y_pixel'] = rna_patch['y_pixel'] - i*img_size
            if len(np.unique(mask_patch))>20:
                mask_patch = Image.fromarray(mask_patch)
                mask_patch.save(os.path.join(output_mask_dir, '{}.tif'.format(id)))
                mask_10x_patch = Image.fromarray(mask_10x_patch)
                mask_10x_patch.save(os.path.join(output_mask_dir, '{}.tif'.format(id)))

                io.imsave(os.path.join(img_seg_dir, '{}.png'.format(id)), seg_img[i*img_size:(i+1)*img_size, j*img_size:(j+1)*img_size])

                dapi_patch = Image.fromarray(dapi_patch)
                dapi_patch.save(os.path.join(output_dapi_dir, '{}.png'.format(id)))

                rna_patch.to_csv(os.path.join(output_rna_dir, '{}.csv'.format(id)), index=False)

                rna_patch['y_pixel'] = img_size - rna_patch['y_pixel']
                rna_patch.plot.scatter(x='x_pixel', y='y_pixel', s=0.5)
                plt.savefig(os.path.join(rna_vis_dir, '{}.png'.format(id)))

                id += 1
                n_cells += len(np.unique(mask_patch))-1

            
    # quit()
print(id)
print(n_cells)