import random
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.applications import VGG19


def save_checkpoints(manager):
    print('Saving model checkpoint...')
    ckpt_path = manager.save()
    print('Saved model checkpoint to %s' % ckpt_path)
    
    
@tf.function
def train_step(model, optimizer, input_images, reference_images, negative_images, input_prompts, reference_prompts):
    with tf.GradientTape() as tape:
        content, perceptual, tv = model.train_batch(input_images, reference_images, negative_images, input_prompts, reference_prompts)
        loss = content + perceptual + tv
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return content, perceptual, tv


# VGG19 encoder
def build_vgg19_encoder(trainable=False):
    vgg19 = VGG19(input_shape=(None, None, 3), include_top=False, weights="imagenet")
    features = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1", "block5_pool"]
    features = [vgg19.get_layer(feature).output for feature in features]
    model = tf.keras.Model(inputs=vgg19.input, outputs=features)
    model.trainable = trainable
    return model


# VGG19 extractor
def build_vgg19_extractor(trainable=False):
    vgg19    = VGG19(input_shape=(None, None, 3), include_top=False, weights="imagenet")
    features = ["block2_conv2", "block3_conv3", "block4_conv3", "block5_conv3"]
    features = [vgg19.get_layer(feature).output for feature in features]
    model = tf.keras.Model(inputs=vgg19.input, outputs=features)
    model.trainable = trainable
    return model


def gram_matrix(activations):
    batch  = tf.shape(activations)[0]
    height = tf.shape(activations)[1]
    width  = tf.shape(activations)[2]
    num_channels = tf.shape(activations)[3]
    gram_matrix  = tf.transpose(activations, [0, 3, 1, 2])
    gram_matrix  = tf.reshape(gram_matrix, [batch, num_channels, width * height])
    gram_matrix  = tf.matmul(gram_matrix, gram_matrix, transpose_b=True)
    return gram_matrix


def expend_as(tensor, rep):
     return tf.keras.layers.Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3), arguments={'repnum': rep})(tensor)


def mean_variance_norm(feat):
    size = tf.shape(feat)
    mean, var = tf.nn.moments(feat, 2)
    std = tf.math.sqrt(var)
    normalized_feat = (feat - tf.broadcast_to(mean, size)) / tf.broadcast_to(std, size)
    return normalized_feat
