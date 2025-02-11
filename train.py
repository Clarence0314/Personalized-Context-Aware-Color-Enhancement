import os
import argparse
import tqdm
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import tensorflow as tf

from dataset import Dataset
from network import AttNet
from utils import save_checkpoints, train_step

# Set Tensorflow to use GPU
devices = tf.config.experimental.list_physical_devices(device_type='GPU')
tf.config.experimental.set_visible_devices(devices=devices[0], device_type='GPU')
tf.config.experimental.set_memory_growth(devices[0], True)


# Set seeds
random.seed(3456)
tf.random.set_seed(3456)


def main(args):
    # Create for tensorboard 
    writer_path = "./logs"
    if not os.path.exists(writer_path): os.makedirs(writer_path)
    writer = tf.summary.create_file_writer(writer_path)
    
    # Load encoded prompts from CSV files
    raw_encoded_prompts = pd.read_csv(args.raw_encoded_prompts)
    expert_encoded_prompts = pd.read_csv(args.expert_encoded_prompts)
    
    # Call dataloader
    dataloader = Dataset(args.src_dir,
                         args.ref_dir,
                         args.neg_dir,
                         args.raw_encoded_prompts,
                         args.expert_encoded_prompts,
                         args.img_size,
                         args.batch_size,
                         args.resize)
    train_iter, val_iter = dataloader()
    
    # Define optimizer and initialize model
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr, amsgrad=True, clipnorm=1.0)
    model = AttNet(use_an=args.use_an)
    
    # Create checkpoint or resume from checkpoint
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
    else:
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), model=model)
        ckpt.restore(tf.train.latest_checkpoint(args.ckpt_dir))
    manager = tf.train.CheckpointManager(ckpt, args.ckpt_dir, max_to_keep=5)
    
    # Training loop
    with writer.as_default():
        for i in tqdm.tqdm(range(args.epochs)):
            input_images, reference_images, negative_images, input_prompts, reference_prompts = next(train_iter)
            content, perceptual, tv = train_step(model, optimizer, input_images, reference_images, negative_images, input_prompts, reference_prompts)
            loss = content + perceptual + tv
            with tf.name_scope("Losses"):
                tf.summary.scalar("Content Loss",    content,    step=i)
                tf.summary.scalar("Perceptual Loss", perceptual, step=i)
                tf.summary.scalar("TV Loss",         tv,         step=i)
                tf.summary.scalar("Total Loss",      loss,       step=i)
            writer.flush()
            ckpt.step.assign_add(1)

            if (i+1) % args.eval_rate == 0:
                input_images, reference_images, negative_images, input_prompts, reference_prompts = next(val_iter)

                with writer.as_default(): 
                    tf.summary.scalar(name='training-validation', data=loss, step=i)

                outputs = model(input_images, reference_images, input_prompts, reference_prompts)

                # Reshape images to match the expected dimensions for tf.summary.image
                input_images = (input_images + 1) * 127.5
                input_images = tf.cast(input_images, tf.uint8)
                input_images = input_images[0]  # Select the first image in the batch

                reference_images = (reference_images + 1) * 127.5
                reference_images = tf.cast(reference_images, tf.uint8)
                reference_images = reference_images[0]  # Select the first image in the batch

                output_images = (outputs + 1) * 127.5
                output_images = tf.cast(output_images, tf.uint8)
                output_images = output_images[0]  # Select the first image in the batch

                # Add a batch dimension
                input_images = tf.expand_dims(input_images, axis=0)
                reference_images = tf.expand_dims(reference_images, axis=0)
                output_images = tf.expand_dims(output_images, axis=0)

                with tf.name_scope("Losses"):
                    tf.summary.image("input_image", input_images, step=i)
                    tf.summary.image("reference_image", reference_images, step=i)
                    tf.summary.image("output_image", output_images, step=i)

            if (i+1) % args.eval_rate == 0 or (i+1) == args.epochs: save_checkpoints(manager)
    print("Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', dest='src_dir', default="Image(Expert_E_256)/raw", help='path to input images')
    parser.add_argument('--ref_dir', dest='ref_dir', default="Image(Expert_E_256)/expert", help='path to reference images')
    parser.add_argument('--neg_dir', dest='neg_dir', default="Image(Expert_E_256)/negatives", help='path to negative images')
    parser.add_argument('--raw_encoded_prompts', dest='raw_encoded_prompts', default='raw_encoded_prompts.csv', help='path to raw encoded prompts CSV')
    parser.add_argument('--expert_encoded_prompts', dest='expert_encoded_prompts', default='expert_encoded_prompts.csv', help='path to expert encoded prompts CSV')
    parser.add_argument('--ckpt_dir', dest='ckpt_dir', default='checkpoints', help='directory for checkpoints')
    parser.add_argument('--epochs', dest='epochs', type=int, default=5000, help='number of total epochs')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='number of samples in one batch')
    parser.add_argument('--img_size', dest='img_size', type=int, default=256, help='image resolution during training')
    parser.add_argument('--lr', dest='lr', type=float, default=1e-4, help='initial learning rate')
    parser.add_argument('--resize', dest='resize', type=bool, default=False, help='whether to resize image')
    parser.add_argument('--eval_rate', dest='eval_rate', type=int, default=10, help='evaluating and saving checkpoints every # epoch')
    parser.add_argument('--use_an',  dest='use_an', type=bool, default=True, help='whether to use weighted BN')
    args = parser.parse_args()

    main(args)
