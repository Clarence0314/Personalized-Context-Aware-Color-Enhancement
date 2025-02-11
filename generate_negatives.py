import os
import glob
import tqdm
import argparse
import cv2
import numpy as np
from colorthief import ColorThief
# import fast_colorthief 

"""_summary_
A script to generate negative images for our triplet-style training.

As our network does not make use of semnatic information, this is a 
relatively cheap way for us to adjust the results. This script will 
extract a dominant color from an image and apply it as an overlay to
the input image. During training, the network will learn to optimize
away from these overlay/negative images with the triplet loss.

If you're on linux, you can use fast-colorthief instead of colorthief,
which will provide a mojor speed up for the operations.

Use it like this:
python generate_negatives.py --src_dir <path-to-groundtruth-images> --dst_dir <destination-path-for-generated-negatives>

"""

def main(args):
    if not os.path.exists(args.dst_dir): os.makedirs(args.dst_dir)
    
    for path in tqdm.tqdm(glob.glob("%s/*"%args.src_dir)):
        name = path.split(os.sep)[-1]
        img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        
        # for colorthief
        color_thief = ColorThief(path)
        dominant_color = color_thief.get_color(quality=1)
        
        # for fast-colorthief
        # dominant_color = fast_colorthief.get_dominant_color(path, quality=1)
        
        r = np.expand_dims(np.full(img.shape[:-1], dominant_color[0]), axis=-1)
        g = np.expand_dims(np.full(img.shape[:-1], dominant_color[1]), axis=-1)
        b = np.expand_dims(np.full(img.shape[:-1], dominant_color[2]), axis=-1)
        dominant_rgb = np.concatenate([r,g,b], axis=-1)
        img_f = np.float32(img) / 255
        dominant_rgb_f = np.float32(dominant_rgb) / 255
        alpha = 0.4
        negative = cv2.addWeighted(img_f.copy(), alpha, dominant_rgb_f.copy(), 1 - alpha, 0, img_f.copy())
        im = cv2.cvtColor(np.uint8(negative*255), cv2.COLOR_RGB2BGR)
        cv2.imwrite('%s/%s'%(args.dst_dir, name), im)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', dest='src_dir', default= "./raw", help='path to input images')
    parser.add_argument('--dst_dir', dest='dst_dir', default="./negatives", help='destination path for the generated negative images')
    args = parser.parse_args()

    main(args)