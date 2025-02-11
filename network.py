import tensorflow as tf
import tensorflow_addons as tfa

from layers import AdaIN
from models import Encoder, Decoder
from utils import gram_matrix, build_vgg19_extractor
from losses import triplet_loss, total_variation_loss


class AttNet(tf.keras.Model):
    def __init__(self, use_an=False):
        super(AttNet, self).__init__()
        self.adain     = AdaIN()
        self.encoder_1 = Encoder(use_an=use_an)
        self.encoder_2 = Encoder(use_an=use_an)
        self.decoder   = Decoder(use_an=use_an)
        self.vgg_extractor = build_vgg19_extractor()

    @tf.function
    def call(self, inputs, references, input_prompts=None, reference_prompts=None, alpha=1., resize=False):
        inter_src, inp_src, skip_conns_src = self.encoder_1(inputs)
        inter_ref, inp_ref, skip_conns_ref = self.encoder_2(references)
        
        # Incorporate the encoded prompts
        if input_prompts is not None and reference_prompts is not None:
            # Cast encoded prompts to float32
            input_prompts = tf.cast(input_prompts, tf.float32)
            reference_prompts = tf.cast(reference_prompts, tf.float32)
            
            # Tile the encoded prompts to match the spatial dimensions
            shape = tf.shape(inter_src)
            batch_size, height, width = shape[0], shape[1], shape[2]
            input_prompts = tf.tile(tf.expand_dims(tf.expand_dims(input_prompts, axis=1), axis=1), [batch_size, height, width, 1])
            reference_prompts = tf.tile(tf.expand_dims(tf.expand_dims(reference_prompts, axis=1), axis=1), [batch_size, height, width, 1])
            
            # Concatenate prompts with intermediate features
            inter_src = tf.concat([inter_src, input_prompts], axis=-1)
            inter_ref = tf.concat([inter_ref, reference_prompts], axis=-1)
        
        inter_ada = self.adain([inter_src, inter_ref, alpha])
        skip_conns_ada = [self.adain([scs, scr, alpha]) for scs, scr in zip(skip_conns_src, skip_conns_ref)]
        skip_conns_ada = [inp_src] + skip_conns_ada

        outputs = self.decoder(inter_ada, skip_conns_ada)
        outputs = tf.clip_by_value(outputs, -1.0, 1.0)
        return outputs

    @tf.function
    def train_batch(self, 
                    inputs, 
                    references, 
                    negatives, 
                    input_prompts, 
                    reference_prompts, 
                    alpha=1.0, 
                    content_loss_weight=10.0,
                    perceptual_loss_weight=1.0,
                    tv_loss_weight=0.0005,
                    **kwargs):
        inter_src, inp_src, skip_conns_src = self.encoder_1(inputs)
        inter_ref, inp_ref, skip_conns_ref = self.encoder_2(references)
        
        # Incorporate the encoded prompts
        if input_prompts is not None and reference_prompts is not None:
            # Cast encoded prompts to float32
            input_prompts = tf.cast(input_prompts, tf.float32)
            reference_prompts = tf.cast(reference_prompts, tf.float32)
            
            # Tile the encoded prompts to match the spatial dimensions
            shape = tf.shape(inter_src)
            batch_size, height, width = shape[0], shape[1], shape[2]
            input_prompts = tf.tile(tf.expand_dims(tf.expand_dims(input_prompts, axis=1), axis=1), [batch_size, height, width, 1])
            reference_prompts = tf.tile(tf.expand_dims(tf.expand_dims(reference_prompts, axis=1), axis=1), [batch_size, height, width, 1])
            
            # Concatenate prompts with intermediate features
            inter_src = tf.concat([inter_src, input_prompts], axis=-1)
            inter_ref = tf.concat([inter_ref, reference_prompts], axis=-1)
        
        inter_ada = self.adain([inter_src, inter_ref, alpha])
        skip_conns_ada = [self.adain([scs,scr,alpha]) for scs,scr in zip(skip_conns_src,skip_conns_ref)]
        skip_conns_ada = [inp_src] + skip_conns_ada

        outputs = self.decoder(inter_ada, skip_conns_ada)
        outputs = tf.clip_by_value(outputs, -1.0, 1.0)
        
        # loss functions
        inp_feats = self.vgg_extractor(inputs)
        out_feats = self.vgg_extractor(outputs)
        ref_feats = self.vgg_extractor(references)
        neg_feats = self.vgg_extractor(negatives)
        
        content   = content_loss_weight * triplet_loss(out_feats[0], inp_feats[0], neg_feats[0], margin=0.5)

        weights    = [50,70,30]
        p          = [w * triplet_loss(x,y,z,margin=0.3) for w,x,y,z in zip(weights,out_feats[1:],ref_feats[1:],neg_feats[1:])]
        perceptual = perceptual_loss_weight * tf.math.reduce_mean(p)
        
        tv = total_variation_loss(outputs, height=outputs.shape[1], width=outputs.shape[2])
        tv = tv_loss_weight * tv

        return content, perceptual, tv
