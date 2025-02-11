import tensorflow as tf


@tf.function
def mse(c_true, c_pred):
    loss = tf.math.reduce_mean(tf.math.square(c_true - c_pred))
    return loss

@tf.function
def triplet_loss(model_anchor, model_positive, model_negative, margin=0.5):
    distance1 = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(model_anchor - model_positive, 2), 1, keepdims=True))
    distance2 = tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(model_anchor - model_negative, 2), 1, keepdims=True))
    return tf.math.reduce_mean(tf.math.maximum(distance1 - distance2 + margin, 0))

@tf.function
def total_variation_loss(x, height, width):
    a = tf.math.square(
        x[:, : height - 1, : width - 1, :] - x[:, 1:, : width - 1, :]
    )
    b = tf.math.square(
        x[:, : height - 1, : width - 1, :] - x[:, : height - 1, 1:, :]
    )
    return tf.math.reduce_sum(tf.math.pow(a + b, 1.25))