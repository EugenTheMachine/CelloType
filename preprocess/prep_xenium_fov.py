import numpy as np
import cv2
import pandas as pd
import os
from skimage import io
import json
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

data_path = 'data/xenium/ffpe_human_lung/output'
save_dir = 'data/xenium_lung'

# data_path = 'data/xenium/ff_mouse_colon/output'
# save_dir = 'data/xenium_mouse_colon'

# data_path = 'data/xenium/ffpe_human_pancreas/output'
# save_dir = 'data/xenium_pancreas'

microns_per_pixel = 0.2125

fov_img_dir = os.path.join(save_dir, 'img')
os.makedirs(fov_img_dir, exist_ok=True)
img_seg_dir = os.path.join(save_dir, 'img_for_seg')
os.makedirs(img_seg_dir, exist_ok=True)
transcript_dir = os.path.join(save_dir, 'transcript')
os.makedirs(transcript_dir, exist_ok=True)
kde_dir = os.path.join(save_dir, 'kde')
os.makedirs(kde_dir, exist_ok=True)
img_dapi_dir = os.path.join(save_dir, 'img_dapi')
os.makedirs(img_dapi_dir, exist_ok=True)
mask_fov_dir = os.path.join(save_dir, 'mask_10x')
os.makedirs(mask_fov_dir, exist_ok=True)

img = io.imread(os.path.join(data_path, 'morphology_focus', 'morphology_focus_0000.ome.tif'))
mask_10x = np.load(os.path.join(data_path, 'cellseg_mask.npy'))
rna = pd.read_csv(os.path.join(data_path, 'transcripts.csv.gz'), compression='gzip')
fov_loc = json.load(open(os.path.join(data_path, 'aux_outputs', 'morphology_fov_locations.json')))

# fov_list = list(rna['fov_name'].unique())
fov_list = fov_loc['fov_locations'].keys()

for fov in tqdm(fov_list):
    # print("Processing {}".format(fov))
    fov_rna = rna[rna['fov_name'] == fov]

    h,w,x,y = fov_loc['fov_locations'][fov].values()
    pix_h, pix_w = int(h/microns_per_pixel), int(w/microns_per_pixel)
    pix_x, pix_y = int(x/microns_per_pixel), int(y/microns_per_pixel)
    img_fov = img[pix_y:pix_y+pix_h, pix_x:pix_x+pix_w, :]
    mask_fov = mask_10x[pix_y:pix_y+pix_h, pix_x:pix_x+pix_w]

    img_seg = np.zeros((pix_h, pix_w, 3), dtype=np.uint8)
    img_seg[:,:,2] = img_fov[:,:,0]//256
    max_fuse = np.max(img_fov[:,:,[1,2,3]], axis=2)
    img_seg[:,:,1] = max_fuse//256
    img_seg = equalize_adapthist(img_seg, clip_limit=0.03)
    img_seg = rescale_intensity(img_seg, out_range=(0, 255))

    img_dapi = img_fov[:,:,0]//256
    img_dapi = equalize_adapthist(img_dapi, clip_limit=0.03)
    img_dapi = rescale_intensity(img_dapi, out_range=(0, 255))
    io.imsave(os.path.join(img_dapi_dir, '{}.png'.format(fov)), img_dapi)

    io.imsave(os.path.join(img_seg_dir, '{}.png'.format(fov)), img_seg)
    io.imsave(os.path.join(fov_img_dir, '{}.tif'.format(fov)), img_fov)
    io.imsave(os.path.join(mask_fov_dir, '{}.tif'.format(fov)), mask_fov)

    fov_rna['x_pixel'] = (fov_rna['x_location'].values - x) / microns_per_pixel
    fov_rna['y_pixel'] = (fov_rna['y_location'].values - y) / microns_per_pixel
    fov_rna.to_csv(os.path.join(transcript_dir, '{}.csv'.format(fov)), index=False)
    # quit()




