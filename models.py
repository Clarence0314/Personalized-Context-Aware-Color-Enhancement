import tensorflow as tf
import tensorflow_addons as tfa
from layers import ResidualBlock, UpConvBlock
from utils import build_vgg19_encoder


class Encoder(tf.keras.Model):
    def __init__(self, inp_ch=3, ngf=64, use_an=False, **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self.vgg   = build_vgg19_encoder()
        self.conv0 = tf.keras.layers.Conv2D(inp_ch, 1, padding="valid", kernel_initializer='glorot_normal', bias_initializer='random_normal', activation="selu")
        self.resb1 = ResidualBlock(ngf, use_an=use_an)
        self.resb2 = ResidualBlock(ngf*2, use_an=use_an)
        self.resb3 = ResidualBlock(ngf*4, use_an=use_an)
        self.resb4 = ResidualBlock(ngf*8, use_an=use_an)
        self.resb5 = ResidualBlock(ngf*8, use_an=use_an)

    @tf.function
    def call(self, inputs, training=False, **kwargs):
        c0 = self.conv0(inputs)
        c1, c2, c3, c4, c5, outputs = self.vgg(c0)
        c1 = self.resb1(c1, bn=training)
        c2 = self.resb2(c2, bn=training)
        c3 = self.resb3(c3, bn=training)
        c4 = self.resb4(c4, bn=training)
        c5 = self.resb5(c5, bn=training)
        return outputs, c0, [c1, c2, c3, c4, c5]


class Decoder(tf.keras.Model):
    def __init__(self, out_ch=3, ngf=64, use_an=False, **kwargs):
        super(Decoder, self).__init__(**kwargs)

        self.ucb1 = UpConvBlock(ngf*8, use_an=use_an)
        self.ucb2 = UpConvBlock(ngf*8, use_an=use_an)
        self.ucb3 = UpConvBlock(ngf*8, use_an=use_an)
        self.ucb4 = UpConvBlock(ngf*4, use_an=use_an)
        self.ucb5 = UpConvBlock(ngf*2, use_an=use_an)
        self.ucb6 = UpConvBlock(ngf, scale=1, use_an=use_an)
        
        self.conv_last = tf.keras.layers.Conv2D(out_ch, 1, padding="valid", kernel_initializer='glorot_normal', bias_initializer='random_normal', activation="tanh")

    @tf.function
    def call(self, inputs, skip_conns, training=False, resize=True, **kwargs):
        c0, c1, c2, c3, c4, c5 = skip_conns
        
        x1 = self.ucb1(inputs,c5, bn=training)
        x2 = self.ucb2(x1,c4, bn=training)
        x3 = self.ucb3(x2,c3, bn=training)
        x4 = self.ucb4(x3,c2, bn=training)
        x5 = self.ucb5(x4,c1, bn=training)
        x6 = self.ucb6(x5,c0, bn=training)

        outputs = self.conv_last(x6)
        return outputs
