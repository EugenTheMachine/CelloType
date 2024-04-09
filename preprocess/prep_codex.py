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

Image.MAX_IMAGE_PIXELS = 1000000000 

img_size = 512

save_dir = '' 
patch_dir = ''
mask_patch_dir = ''
mask_dir = ''
crc_a_dir = ''
crc_b_dir = ''

# os.mkdir(mask_patch_dir)

metadata = pd.read_csv('', index_col=0)
unique_regions = list(metadata['Region'].unique())
unique_nb = list(metadata['ClusterName'].unique())

tr_index = [str(i).zfill(3) for i in range(1, 61)]
test_index = [str(i).zfill(3) for i in range(61, 71)]

# remove nan
unique_nb = [str(i) for i in unique_nb if str(i) != 'nan']
nb2num = {unique_nb[i]: i for i in range(len(unique_nb))}
with open(save_dir+'/ct2num.json', 'w') as fp:
    json.dump(nb2num, fp)

dataset_dicts = []
img_id = 0

for stage in ['test']:
# for stage in ['train']:
    for tma in ['A','B']:
        print("Processing {}_{}".format(tma, stage))
        crc_dir = crc_a_dir if tma == 'A' else crc_b_dir
        reg_list = tr_index if stage == 'train' else test_index
        codex_files = os.listdir(crc_dir)
        for reg_id in tqdm(reg_list):
            # print("Processing {}_{}".format(tma, img_id))
            codex_file = [i for i in codex_files if i.startswith('reg'+reg_id)][0]
            cell_file = [i for i in os.listdir(mask_dir) if i.startswith('CRC_'+tma+'_reg'+reg_id)][0]
            cell_mask = io.imread(os.path.join(mask_dir, cell_file))
            region = 'reg'+reg_id

            codex_img = io.imread(os.path.join(crc_dir, codex_file))
            # codex_img = np.transpose(codex_img, (1,2,3,0))
            # codex_img = np.reshape(codex_img, (codex_img.shape[0], codex_img.shape[1], -1))
            # codex_img = codex_img//2**8
            # codex_img = codex_img.astype(np.uint8)
            # io.imsave(os.path.join(crc_dir, codex_file), codex_img)

            w = codex_img.shape[0]//img_size
            h = codex_img.shape[1]//img_size

            region_meta = metadata[metadata['Region']==region]
            region_meta = region_meta[region_meta['TMA_AB']==tma][['X:X','Y:Y','ClusterName']]

            for w_i in range(w):
                for h_i in range(h):
                    codex_patch = codex_img[w_i*img_size:(w_i+1)*img_size, h_i*img_size:(h_i+1)*img_size, :]
                    mask_patch = cell_mask[w_i*img_size:(w_i+1)*img_size, h_i*img_size:(h_i+1)*img_size]

                    io.imsave(os.path.join(patch_dir, '{}_{}_{}_{}.tif'.format(tma, region, w_i, h_i)), codex_patch)
                    io.imsave(os.path.join(mask_patch_dir, '{}_{}_{}_{}.tif'.format(tma, region, w_i, h_i)), mask_patch)

                    metadata_patch = region_meta.copy()
                    metadata_patch['X:X'] = metadata_patch['X:X'] - h_i*img_size
                    metadata_patch['Y:Y'] = metadata_patch['Y:Y'] - w_i*img_size
                    # remove the cells outside the patch
                    metadata_patch = metadata_patch[metadata_patch['X:X']>=0]
                    metadata_patch = metadata_patch[metadata_patch['Y:Y']>=0]
                    metadata_patch = metadata_patch[metadata_patch['X:X']<img_size]
                    metadata_patch = metadata_patch[metadata_patch['Y:Y']<img_size]

                    record = {}
                    record['file_name'] = os.path.join(patch_dir, '{}_{}_{}_{}.tif'.format(tma, region, w_i, h_i))
                    record['image_id'] = img_id
                    record['height'] = img_size
                    record['width'] = img_size
                    objs = []

                    match_list = []
                    for t in range(len(metadata_patch)):
                        coord = metadata_patch.iloc[t][['Y:Y','X:X']].values
                        nb = str(metadata_patch.iloc[t]['ClusterName'])
                        k = mask_patch[coord[0], coord[1]]
                        if k != 0 and k not in match_list and nb in nb2num.keys():
                            match_list.append(k)
                            obj = {}
                            obj['bbox'] = np.array(np.where(mask_patch==k)).min(axis=1)[[1,0]].tolist() + np.array(np.where(mask_patch==k)).max(axis=1)[[1,0]].tolist()
                            obj['bbox_mode'] = 0
                            obj['category_id'] = nb2num[nb]
                            # obj['category_id'] = 0
                            contours, _ = cv2.findContours((mask_patch==k).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                            if len(contours) ==1:
                                contour = contours[0]
                            else:
                                contours_lens = [len(contours[i]) for i in range(len(contours))]
                                contour = contours[np.argmax(contours_lens)]
                            obj['segmentation'] = [np.array(contour).flatten().tolist()]

                            if len(contour) > 5:
                                objs.append(obj)

                    record['annotations'] = objs
                    dataset_dicts.append(record)
                    img_id += 1

    print("Total {} images".format(img_id))
    np.save(os.path.join(save_dir, 'dataset_dicts_patch_{}_cell.npy'.format(stage)), dataset_dicts)

        


