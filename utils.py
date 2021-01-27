import numpy as np
import math
from PIL import Image
import tqdm
from tensorflow.python.keras import backend as K
import os
import matplotlib.pyplot as plt
from skimage.measure import find_contours
from matplotlib.patches import Polygon
    

def prediction(model, img_dir, out_dir,input_shape = (224,224)):
    
    imgs = os.listdir(img_dir)

    for img in tqdm.tqdm(imgs):
        
        img_path = os.path.join(img_dir,img)
        img_arr = np.array(Image.open(img_path).resize(input_shape))
        img_arr = img_arr.reshape((1,)+img_arr.shape)/255.

        pre_mask, pre_MS = model.predict(img_arr)

        maskImageGen(img_arr[0], pre_mask[0], pre_MS[0][0], img, out_dir)

def maskImageGen(arr,mask,label,img_name,out_dir = '.'):

    #set imag
    img = arr * 255
    mask_label = mask[...,0] >= 0.5
    color = (0,1,0.2)
    alpha = 0.3
    
    #drawing mask
    for c in range(3):
        img[:, :, c] = np.where(mask[...,0] >= 0.5,
                                img[:, :, c] * (1 - alpha)
                                + alpha * color[c] * 255,
                                img[:, :, c])

    #make empty plot
    _, ax = plt.subplots(1, figsize=(4,4))
    height, width = mask.shape[:-1]
    ax.set_ylim(height, 0)
    ax.set_xlim(0, width)
    ax.axis('off')

    #Segmenation Area
    x = mask.shape[1]
    y = 0
    _ = ax.text(x - 90, y + 50, str(round(label,1)),
                color='w', size=64, backgroundcolor="none")

    #Drawing Mask edge
    padded_mask = np.zeros((mask_label.shape[0] + 2, mask_label.shape[1] + 2), dtype=np.uint8)
    padded_mask[1:-1, 1:-1] = mask_label
    contours = find_contours(padded_mask, 0.5)

    for verts in contours:
        # Subtract the padding and flip (y, x) to (x, y)
        verts = np.fliplr(verts) - 1
        p = Polygon(verts, facecolor="none", edgecolor=color)
        ax.add_patch(p)
    
    #Save results Image
    ax.imshow(img.astype(np.uint8))
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir,img_name))
    plt.close()

