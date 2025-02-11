import glob
import tqdm
import numpy as np
from PIL import Image
import pandas as pd
import os
import tensorflow as tf

class Dataset:
    def __init__(self,
                 src_path,
                 ref_path,
                 neg_path,
                 raw_encoded_prompts_path,
                 expert_encoded_prompts_path,
                 img_size=128,
                 batch_size=2,
                 resize=False,
                 augmentations=False):
        super(Dataset, self).__init__()
        self.img_size = img_size
        self.batch_size = batch_size
        self.resize = resize
        self.augmentations = augmentations

        srcs = sorted(glob.glob("%s/*.jpg"%src_path))
        refs = sorted(glob.glob("%s/*.jpg"%ref_path))
        negs = sorted(glob.glob("%s/*.jpg"%neg_path))
        
        self.srcs = srcs
        self.refs = refs
        self.negs = negs
        
        # Load encoded prompts
        self.raw_encoded_prompts = pd.read_csv(raw_encoded_prompts_path)
        self.expert_encoded_prompts = pd.read_csv(expert_encoded_prompts_path)
        
        # Create dictionaries for quick lookup
        self.raw_prompt_dict = dict(zip(self.raw_encoded_prompts['image'], self.raw_encoded_prompts['encoded_prompt'].apply(eval)))
        self.expert_prompt_dict = dict(zip(self.expert_encoded_prompts['image'], self.expert_encoded_prompts['encoded_prompt'].apply(eval)))

    def list_splitter(self, list_to_split, ratio=0.8):
        elements = len(list_to_split)
        middle = int(elements * ratio)
        return [list_to_split[:middle], list_to_split[middle:]]

    def __len__(self):
        assert (len(self.src_imgs) == len(self.ref_imgs)) \
                and (len(self.src_imgs) == len(self.neg_imgs))
        return len(self.src_imgs)

    def __call__(self):
        src_imgs = []
        ref_imgs = []
        neg_imgs = []
        src_prompts = []
        ref_prompts = []
        
        for idx in tqdm.tqdm(range(len(self.srcs))):
            src_img = Image.open(self.srcs[idx])
            ref_img = Image.open(self.refs[idx])
            neg_img = Image.open(self.negs[idx])
            
            # Resize images if required
            if self.resize:
                src_img = src_img.resize((self.img_size, self.img_size), Image.BICUBIC)
                ref_img = ref_img.resize((self.img_size, self.img_size), Image.BICUBIC)
                neg_img = neg_img.resize((self.img_size, self.img_size), Image.BICUBIC)

            # Normalize images
            src_img = (np.array(src_img).astype(np.float32) / 127.5) - 1
            ref_img = (np.array(ref_img).astype(np.float32) / 127.5) - 1
            neg_img = (np.array(neg_img).astype(np.float32) / 127.5) - 1
            
            src_img = np.clip(src_img, -1., 1.)
            ref_img = np.clip(ref_img, -1., 1.)
            neg_img = np.clip(neg_img, -1., 1.)

            if not np.isnan(np.sum(src_img)) or \
               not np.isnan(np.sum(ref_img)) or \
               not np.isnan(np.sum(neg_img)):
                src_imgs.append(src_img)
                ref_imgs.append(ref_img)
                neg_imgs.append(neg_img)
                
                # Get encoded prompts
                img_name = os.path.basename(self.srcs[idx])
                src_prompts.append(self.raw_prompt_dict[img_name])
                ref_prompts.append(self.expert_prompt_dict[img_name])
                
        train_src, val_src = self.list_splitter(src_imgs, ratio=0.8)
        train_ref, val_ref = self.list_splitter(ref_imgs, ratio=0.8)
        train_neg, val_neg = self.list_splitter(neg_imgs, ratio=0.8)
        train_src_prompts, val_src_prompts = self.list_splitter(src_prompts, ratio=0.8)
        train_ref_prompts, val_ref_prompts = self.list_splitter(ref_prompts, ratio=0.8)
        
        tf_train_data = tf.data.Dataset.from_tensor_slices(
                                        (np.asarray(train_src),
                                         np.asarray(train_ref),
                                         np.asarray(train_neg),
                                         np.asarray(train_src_prompts),
                                         np.asarray(train_ref_prompts)))
        tf_val_data = tf.data.Dataset.from_tensor_slices(
                                        (np.asarray(val_src),
                                         np.asarray(val_ref),
                                         np.asarray(val_neg),
                                         np.asarray(val_src_prompts),
                                         np.asarray(val_ref_prompts)))

        tf_train_data = tf_train_data.repeat().shuffle(30).batch(self.batch_size)
        tf_val_data = tf_val_data.repeat().shuffle(30).batch(self.batch_size)
        
        if self.augmentations: 
            print("Perform augmentations ---")
            trainAug = tf.keras.Sequential([
                tf.keras.layers.experimental.preprocessing.Rescaling(scale=1.0 / 255),
                tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
                tf.keras.layers.experimental.preprocessing.RandomZoom(
                    height_factor=(-0.05, -0.15),
                    width_factor=(-0.05, -0.15)),
                tf.keras.layers.experimental.preprocessing.RandomRotation(0.3)
            ])
            
            tf_train_data = tf_train_data.map(lambda x,y,z,p,q: 
                (trainAug(x),y,z,p,q), num_parallel_calls=tf.data.experimental.AUTOTUNE)
                                        
        tf_train_data = tf_train_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        tf_val_data = tf_val_data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_iter = iter(tf_train_data)
        val_iter = iter(tf_val_data)

        return train_iter, val_iter
