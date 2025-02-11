import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.keras.backend as K
from utils import expend_as, mean_variance_norm


"""Adaptive Instance Normalization
Adaptive Instance Normalization is a normalization method 
that aligns the mean and variance of the content features 
with those of the style features. Instance Normalization 
normalizes the input to a single style specified by the 
affine parameters. Adaptive Instance Normalization is an 
extension. In AdaIN, we receive a content input and a style 
input, and we simply align the channel-wise mean and variance 
of x to match those of y. Unlike Batch Normalization, 
Instance Normalization or Conditional Instance Normalization, 
AdaIN has no learnable affine parameters.

Idea from 
https://arxiv.org/abs/1703.06868

Implementation taken from 
https://github.com/rasmushr97/AdaIN-style-transfer---Tensorflow-2.0/blob/master/adain.py

Input Parameters:
    Takes a list of parameters when you call it
    list:
        content: input image or feature map (B x H x W x C)
        style: reference image or feature map
        alpha: controls the rate of interpolation between 
               content and style (input and ref), 
               defaults to 1.
Returns:
    An output image/feature, same size as input content and style
"""
class AdaIN(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        self.eps = epsilon
        super(AdaIN, self).__init__(**kwargs)

    def get_config(self):
        config = super().get_config().copy()
        config.update({'epsilon': self.eps})
        return config

    def call(self, inputs):
        content, style, alpha = inputs
        axes = [1, 2]
        c_mean, c_var = tf.nn.moments(content, axes=axes, keepdims=True)
        s_mean, s_var = tf.nn.moments(style, axes=axes, keepdims=True)
        c_std, s_std = tf.math.sqrt(
            c_var + self.eps), tf.math.sqrt(s_var + self.eps)
        adain = s_std * (content - c_mean) / c_std + s_mean
        return alpha * adain + (1-alpha) * content


"""UpConvBlock
Upsampling block consisting of SE and AG modules, 
with Batch Attentive Normalization layers. Used in
the decoder network to reconstruct the final output

Input parameters:
    inputs: input feature map (B x H x W x C)
    sc: feature maps from previous layers, 
        used as skip connection

Returns:
    An output tensor with the same size as input
"""
class UpConvBlock(tf.keras.layers.Layer):
    def __init__(self, filter, kernel=3, strides=1, scale=2, use_an=False):
        super(UpConvBlock,self).__init__()
        self.use_an = use_an

        self.sqex1 = SqueezeExciteBlock()
        self.gate1 = GatingSignalLayer(filter)
        self.attb1 = AttentionBlock(filter, scale=2)
        self.ups1  = tf.keras.layers.UpSampling2D(size=scale, interpolation='bilinear')
        self.conv1 = tf.keras.layers.Conv2D(filter, kernel, padding="same", kernel_initializer='glorot_normal', bias_initializer='random_normal')
        self.acti1 = tf.keras.layers.Activation('selu')  #tfa.layers.GELU()
        self.bn1   = BatchAttNorm() if use_an else tf.keras.layers.BatchNormalization()
        self.dol   = tf.keras.layers.Dropout(0.7)
        self.resb1 = ResidualBlock(filter)

    @tf.function
    def call(self, inputs, sc, bn=False):
        x = self.sqex1(inputs, bn=bn)
        x = tf.concat([inputs,x], axis=-1)
        gx = self.gate1(x, bn=bn)
        x = self.attb1(x, gx, bn=bn)
        x = self.ups1(x)
        x = self.conv1(x)
        x = self.acti1(x)
        
        if self.use_an: x = self.bn1(x)
        else: x = self.bn1(x) if bn else x
            
        # x = self.dol(x) if bn else x
        x = tf.concat([x,sc], axis=-1)
        x = self.resb1(x, bn=bn)
        return x


"""SqueezeExciteBlock
Squeeze-and-excitation blocks re-calibrate feature maps by utilising 
a "squeeze" operation of global average pooling followed by a "excitation" 
operation using two completely linked layers. Squeeze-and-excitation blocks 
may be integrated into any CNN architecture/ structure and have a low 
computational cost.

Idea from 
https://arxiv.org/abs/1709.01507

Implementation referred from 
https://github.com/mehul-k5/Double-Unet/blob/009fd186f95a02f772c8bf405897e1eb01137644/model.py

Input parameters:
    inputs: input feature map (B x H x W x C)
    bn: flag to disable standard BN during inference

Returns:
    An output tensor with the same size as input
"""
class SqueezeExciteBlock(tf.keras.layers.Layer):
    def __init__(self, ratio=8, axis=-1, use_an=False, **kwargs):
        super(SqueezeExciteBlock, self).__init__(**kwargs)
        self.use_an = use_an

        self.ratio = ratio
        self.axis = axis
        self.bn1 = BatchAttNorm() if use_an else tf.keras.layers.BatchNormalization()
        self.bn2 = BatchAttNorm() if use_an else tf.keras.layers.BatchNormalization()

    def build(self, input_shape):
        super(SqueezeExciteBlock, self).build(input_shape)
        self.GlobalAvgPooling = tf.keras.layers.GlobalAveragePooling2D()
        self.GlobalAvgPooling.build(input_shape)
        self.filters = input_shape[self.axis]
        self.dense1 = tf.keras.layers.Dense(
            self.filters // self.ratio, activation=tf.keras.layers.Activation('relu'), use_bias=False)
        self.dense2 = tf.keras.layers.Dense(
            self.filters, activation='sigmoid', use_bias=False)

    @tf.function
    def call(self, inputs, bn=False, **kwargs):
        avg = self.GlobalAvgPooling(inputs)
        se_shape = (inputs.shape[0], 1, 1, self.filters)
        se = tf.reshape(avg, se_shape)
        
        if self.use_an: se = self.bn1(x)
        else: se = self.bn1(se) if bn else se
        
        d1 = self.dense1(se)
        
        if self.use_an: d1 = self.bn2(d1)
        else: d1 = self.bn2(d1) if bn else d1
        
        d2 = self.dense2(d1)
        x = tf.multiply(inputs, d2)
        return x


"""GatingSignalLayer
Resize the down layer feature map into the same dimension 
as the up layer feature map using 1x1 conv

Implementation referred from 
https://github.com/MoleImg/Attention_UNet/blob/master/AttSEResUNet.py

Input Parameters:
    inputs: input down-dim feature map (B x H x W x C)

Returns:
    The gating feature map with the same dimension of the up layer feature map
"""
class GatingSignalLayer(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_an=False, **kwargs):
        super(GatingSignalLayer, self).__init__(**kwargs)
        self.use_an = use_an

        self.conv = tf.keras.layers.Conv2D(output_dim, 1, padding='valid')
        self.activation = tf.keras.layers.Activation('relu')
        self.bn = BatchAttNorm() if use_an else tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs, bn=False, **kwargs):
        x = self.conv(inputs)
        x = self.activation(x)
        
        if self.use_an: x = self.bn(x)
        else: x = self.bn(x) if bn else x
        
        return x


"""AttentionBlock
Self gated attention, attention mechanism on spatial dimension

Idea from 
https://www.sciencedirect.com/science/article/pii/S1361841518306133

Implementation referred from 
https://github.com/MoleImg/Attention_UNet/blob/master/AttSEResUNet.py

Input Parameters:
    inputs: input feature map
    gating_inputs: output from GatingSignalLayer, 
                   feature map from the lower layer

Returns:
    Attention weighted on spatial dimension feature map
"""
class AttentionBlock(tf.keras.layers.Layer):
    def __init__(self, inter_dim, scale=2, use_an=False, **kwargs):
        super(AttentionBlock, self).__init__(**kwargs)
        self.use_an = use_an

        self.inter_dim = inter_dim
        self.add = tf.keras.layers.Add()
        self.activation = tf.keras.layers.Activation('relu')
        self.sigmoid = tf.keras.layers.Activation('sigmoid')

        self.conv1 = tf.keras.layers.Conv2D(inter_dim, 2, strides=2, padding="same")
        self.conv2 = tf.keras.layers.Conv2D(inter_dim, 2, strides=2, padding="same")

        self.ups1 = tf.keras.layers.UpSampling2D((1,1), interpolation='bilinear')
        self.conv3 = tf.keras.layers.Conv2D(inter_dim, 1, strides=1, padding="same")

        self.conv4 = tf.keras.layers.Conv2D(1, 1, strides=1, padding="same")
        self.ups2 = tf.keras.layers.UpSampling2D((2,2), interpolation='bilinear')

        self.conv5 = tf.keras.layers.Conv2D(inter_dim, 1, strides=1, padding="same")

        self.bn = BatchAttNorm() if use_an else tf.keras.layers.BatchNormalization()

    @tf.function
    def call(self, inputs, gating_inputs, bn=False, **kwargs):
        shape_x, shape_g = K.int_shape(inputs), K.int_shape(gating_inputs)
        theta_x = self.conv1(inputs)
        shape_theta_x = K.int_shape(theta_x)
        phi_g = self.conv2(gating_inputs)
        upsample_g = self.ups1(phi_g)
        upsample_g = self.conv3(upsample_g)
        concat_xg = self.add([upsample_g, theta_x])
        act_xg = self.activation(concat_xg)
        psi = self.conv4(act_xg)
        sigmoid_xg = self.sigmoid(psi)
        shape_sigmoid = K.int_shape(sigmoid_xg)
        upsample_psi = self.ups2(sigmoid_xg)
        upsample_psi = expend_as(upsample_psi, shape_x[3])
        y = tf.multiply(upsample_psi, inputs)
        result = self.conv5(y)
        
        if self.use_an: result = self.bn(result)
        else: result = self.bn(result) if bn else result
        
        return result


"""ResidualBlock
A residual block is a stack of layers set in such a way 
that the output of a layer is taken and added to another 
layer deeper in the block. The non-linearity is then applied 
after adding it together with the output of the corresponding 
layer in the main path.

Idea from
https://arxiv.org/abs/1512.03385

Input Parameters:
    inputs: input feature map (B x H x W x C)

Returns:
    output feature map with the same dimensions as the input feature map
"""
class ResidualBlock(tf.keras.layers.Layer):
    def __init__(self, filters=32, strides=1, use_an=False, **kwargs):
        super(ResidualBlock, self).__init__(**kwargs)
        self.use_an = use_an

        self.conv0 = tf.keras.layers.Conv2D(filters, kernel_size=1, strides=strides, padding='valid')
        self.conv1 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')
        self.acti1 = tf.keras.layers.Activation('selu')
        self.bn1 = BatchAttNorm() if use_an else tf.keras.layers.BatchNormalization()
        self.dol1 = tf.keras.layers.Dropout(0.7)
        
        self.conv2 = tf.keras.layers.Conv2D(filters, kernel_size=3, strides=1, padding='same')
        self.acti2 = tf.keras.layers.Activation('selu')
        self.bn2 = BatchAttNorm() if use_an else tf.keras.layers.BatchNormalization()
        self.dol2 = tf.keras.layers.Dropout(0.7)
        self.add = tf.keras.layers.Add()

    @tf.function
    def call(self, inputs, bn=False, **kwargs):
        inputs = self.conv0(inputs)
        x = self.conv1(inputs)
        x = self.acti1(x)
        
        if self.use_an: x = self.bn1(x)
        else: x = self.bn1(x) if bn else x
        
        x = self.conv2(x)
        x = self.acti2(x)
        
        if self.use_an: x = self.bn2(x)
        else: x = self.bn2(x) if bn else x
        
        x = self.add([x, inputs])
        return x


"""BatchAttNorm 
Attentive Normalization generalizes the common affine transformation 
component in the vanilla feature normalization. Instead of learning 
a single affine transformation, AN learns a mixture of affine 
transformations and utilizes their weighted-sum as the final affine 
transformation applied to re-calibrate features in an instance-specific 
way. The weights are learned by leveraging feature attention.

Idea from 
https://arxiv.org/abs/1908.01259

Iplementation from 
https://github.com/Cyril9227/EfficientMixNet/blob/34d50152d3894c0c7b5175d43e42c72a04c49f19/keras_efficientmixnets/custom_batchnorm.py

Input Parameters:
    inputs: input feature map (B x H x W x C)

Returns:
    output feature map with the same dimensions as the input feature map
"""
class BatchAttNorm(tf.keras.layers.BatchNormalization):
    def __init__(self, momentum=0.99, epsilon=0.001, axis=-1, **kwargs):
        super(BatchAttNorm, self).__init__(momentum=momentum,
                                           epsilon=epsilon, axis=axis, center=False, scale=False, **kwargs)

    def build(self, input_shape):
        if len(input_shape) != 4:
            raise ValueError(
                'expected 4D input (got {}D input)'.format(input_shape))
        super(BatchAttNorm, self).build(input_shape)

        self.GlobalAvgPooling = tf.keras.layers.GlobalAveragePooling2D(
            "channels_last")
        self.GlobalAvgPooling.build(input_shape)

        self.weight = self.add_weight(
            name='weight', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(1), trainable=True)
        self.bias = self.add_weight(
            name='bias', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(0), trainable=True)
        self.weight_readjust = self.add_weight(
            name='weight_readjust', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(0), trainable=True)
        self.bias_readjust = self.add_weight(
            name='bias_readjust', shape=input_shape[-1], initializer=tf.keras.initializers.Constant(-1), trainable=True)

    def call(self, inputs):
        avg = self.GlobalAvgPooling(inputs)
        attention = K.sigmoid(avg * self.weight_readjust + self.bias_readjust)
        bn_weights = self.weight * attention
        out_bn = super(BatchAttNorm, self).call(inputs)

        if K.int_shape(inputs)[0] is None or K.int_shape(inputs)[0] > 1:
            bn_weights = bn_weights[:, None, None, :]
        return out_bn * bn_weights + self.bias


class ReflectionPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], 'REFLECT')


class SymmetricPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=1, **kwargs):
        self.padding = tuple(padding)
        super(SymmetricPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2 * self.padding[1], input_shape[3])

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        return tf.pad(input_tensor, [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]], 'SYMMETRIC')
