import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from tqdm import tqdm
import cv2
import json

npz_dir = ''

output_dir = '../data'


def get_cell_dicts(data_X, data_y, img_dir, d='train'):
    dataset_dicts = []

    for i in tqdm(range(len(data_X))):
        record = {}
        example = data_X[i]
        example = np.concatenate([np.zeros_like(example[:,:,0:1]),example], axis=2)
        example = example[:,:,[0,2,1]]
        example = rescale_intensity(example, out_range=(0.0, 1.0))
        example = equalize_adapthist(example, kernel_size=None)
        example_save = example*255
        example_save = example_save.astype(np.uint8)[:, :, ::-1]
        cv2.imwrite(os.path.join(img_dir, 'X_{}.png'.format(i)), example_save)

        record['file_name'] = os.path.join(img_dir, 'X_{}.png'.format(i))
        record['height'] = example.shape[0]
        record['width'] = example.shape[1]
        record['image_id'] = i

        objs = []
        mask = data_y[i,:,:,1]

        for j in np.unique(mask)[1:]:
            obj = {}
            obj['bbox'] = np.array(np.where(mask==j)).min(axis=1)[[1,0]].tolist() + np.array(np.where(mask==j)).max(axis=1)[[1,0]].tolist()
            obj['bbox_mode'] = 0
            obj['category_id'] = 0
            contours, _ = cv2.findContours((mask==j).astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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
    
    return dataset_dicts

for d in ['test', 'val', 'train']:

    print("Processing {}".format(d))

    data_dict = np.load(os.path.join(npz_dir, 'tissuenet_v1.0_{}.npz'.format(d)))
    data_dict_X, data_dict_y = data_dict['X'], data_dict['y']
    
    img_dir = os.path.join(output_dir, d)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    dataset_dicts = get_cell_dicts(data_dict_X, data_dict_y, img_dir, d)

    np.save(os.path.join(output_dir, 'dataset_dicts_nuclear_{}.npy'.format(d)), dataset_dicts)
    print("number of images: {}".format(len(dataset_dicts)))