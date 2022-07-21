"""
An implementation of the Set Transformers framework paper in Tensorflow/Keras.

The paper can be found here: https://arxiv.org/abs/1810.00825
The official pytorch implementation can be at: https://github.com/juho-lee/set_transformer
"""

from collections import defaultdict
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K

# Utility Functions --------------------------------------------------------------------------------

def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate

@static_vars(layers={})
def CustomLayer(Layer):
    """
    A decorator for keeping track of custom layers
    """
    CustomLayer.layers[Layer.__name__] = Layer
    return Layer

def custom_layers():
    """
    Fetch custom layer instances
    """
    layers = CustomLayer.layers.copy()
    return layers

# Layer Utility Functions --------------------------------------------------------------------------

# def spectral_norm(*args, **kwargs):
    # import tensorflow_addons as tfa
    # return tfa.layers.SpectralNormalization(*args, **kwargs)
    # return DenseSN(*args, **kwargs)

def spectral_dense(dim, use_spectral_norm=False, activation=None, **kwargs):
    if not use_spectral_norm or activation is not None:
        return keras.layers.Dense(dim, activation=activation, **kwargs)
    return DenseSN(dim, **kwargs)

# Layer Definitions --------------------------------------------------------------------------------

class DenseSN(keras.layers.Dense):
    """
    https://github.com/IShengFang/SpectralNormalizationKeras/blob/master/SpectralNormalizationKeras.py#L17-L69
    Spectral Normalization Layer
    """
    def build(self, input_shape):
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.u = self.add_weight(shape=tuple([1, self.kernel.shape.as_list()[-1]]),
                                 initializer=keras.initializers.RandomNormal(0, 1),
                                 name='sn',
                                 trainable=False)
        self.input_spec = keras.layers.InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True
        
    def call(self, inputs, training=None):
        def _l2normalize(v, eps=1e-12):
            return v / (K.sum(v ** 2) ** 0.5 + eps)
        def power_iteration(W, u):
            _u = u
            _v = _l2normalize(K.dot(_u, K.transpose(W)))
            _u = _l2normalize(K.dot(_v, W))
            return _u, _v
        W_shape = self.kernel.shape.as_list()
        #Flatten the Tensor
        W_reshaped = K.reshape(self.kernel, [-1, W_shape[-1]])
        _u, _v = power_iteration(W_reshaped, self.u)
        #Calculate Sigma
        sigma=K.dot(_v, W_reshaped)
        sigma=K.dot(sigma, K.transpose(_u))
        #normalize it
        W_bar = W_reshaped / sigma
        #reshape weight tensor
        if training in {0, False}:
            W_bar = K.reshape(W_bar, W_shape)
        else:
            with tf.control_dependencies([self.u.assign(_u)]):
                 W_bar = K.reshape(W_bar, W_shape)  
        output = K.dot(inputs, W_bar)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output

@CustomLayer
class VaswaniMultiHeadAttention(keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, use_spectral_norm=False, **kwargs):
        super().__init__(**kwargs)

        assert embed_dim % num_heads == 0, "Embed dim must be divisible by the number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.use_spectral_norm = use_spectral_norm

        self.fc_q = spectral_dense(embed_dim, use_spectral_norm)
        self.fc_k = spectral_dense(embed_dim, use_spectral_norm)
        self.fc_v = spectral_dense(embed_dim, use_spectral_norm)


    def call(self, q, v, k=None, training=None):
        """
        Compute multi-head attention in exactly the same manner
        as the official implementation.

        Reference: https://github.com/juho-lee/set_transformer/blob/master/modules.py#L20-L33
        """
        if k is None:
            k = v
        q, k, v = self.fc_q(q), self.fc_k(k), self.fc_v(v)

        # Divide for multi-head attention
        q_split = tf.concat(tf.split(q, self.num_heads, 2), 0)
        k_split = tf.concat(tf.split(k, self.num_heads, 2), 0)
        v_split = tf.concat(tf.split(v, self.num_heads, 2), 0)

        # Compute attention
        att = tf.nn.softmax(tf.matmul(q_split, k_split, transpose_b=True)/np.sqrt(self.embed_dim), 2)
        out = tf.concat(tf.split(tf.matmul(att, v_split), self.num_heads, 0), 2)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "use_spectral_norm": self.use_spectral_norm
        })
        return config


@CustomLayer
class MultiHeadAttentionBlock(keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        ff_dim=None,
        ff_activation="relu",
        use_layernorm=True,
        pre_layernorm=False,
        is_final_block=False,
        use_keras_mha=True,
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = embed_dim if ff_dim is None else ff_dim
        self.ff_activation = ff_activation
        self.use_layernorm = use_layernorm
        self.pre_layernorm = pre_layernorm
        self.is_final_block = is_final_block
        self.use_keras_mha = use_keras_mha
        self.use_spectral_norm = use_spectral_norm

        # Attention layer
        if self.use_keras_mha:
            self.att = keras.layers.MultiHeadAttention(
                key_dim=self.embed_dim,
                num_heads=self.num_heads)
        else:
            self.att = VaswaniMultiHeadAttention(
                self.embed_dim,
                self.num_heads,
                self.use_spectral_norm)

        # Feed-forward layer
        self.ffn = keras.Sequential([
            keras.layers.Dense(self.ff_dim, activation=self.ff_activation),
            spectral_dense(self.embed_dim, self.use_spectral_norm)])

        if self.use_layernorm:
            self.layernorms = defaultdict(lambda: keras.layers.LayerNormalization(epsilon=1e-6))


    def call_pre_layernorm(self, x, y, training=None):
        x_norm = self.layernorms['x'](x)
        y_norm = self.layernorms['y'](y) if y is not x else x_norm

        # Multi-head attention
        attn = x + self.att(x_norm, y_norm, y_norm, training=training)

        # ff-projection
        out = self.layernorms["attn"](attn)
        out = attn + self.ffn(out)

        if self.is_final_block:
            out = self.layernorms["final"](out)
        return out


    def call_post_layernorm(self, x, y, training=None):
        # Multi-head attention
        attn = x + self.att(x, y, y, training=training)
        if self.use_layernorm:
            attn = self.layernorms["attn"](attn)

        # ff-projection
        out = attn + self.ffn(attn)
        if self.use_layernorm:
            out = self.layernorms["final"](out)
        return out
    
    def compute_mask(self, inputs, mask):
        return mask

    def call(self, inputs: tuple, training=None):
        x, y = inputs[0], inputs[1]
        
        if self.use_layernorm and self.pre_layernorm:
            return self.call_pre_layernorm(x, y, training)
        return self.call_post_layernorm(x, y, training)


    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "ff_activation": self.ff_activation,
            "use_layernorm": self.use_layernorm,
            "pre_layernorm": self.pre_layernorm,
            "is_final_block": self.is_final_block,
            "use_keras_mha": self.use_keras_mha,
            "use_spectral_norm": self.use_spectral_norm
        })
        return config


@CustomLayer
class SetAttentionBlock(MultiHeadAttentionBlock):
    def call(self, inputs, training=None):
        
        x = inputs[0]
        mems = inputs[1]
        
        if mems is not None:
            mems = tf.concat((mems, x),1)
        else:
            mems = x
            
        return super().call([x, mems], training=training)


@CustomLayer
class InducedSetAttentionBlock(keras.layers.Layer):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_induce,
        ff_dim=None,
        ff_activation="relu",
        use_layernorm=True,
        pre_layernorm=False,
        is_final_block=False,
        use_keras_mha=True,
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_induce = num_induce
        self.use_spectral_norm = use_spectral_norm

        self.mab1 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, use_layernorm,
            pre_layernorm, False, use_keras_mha, use_spectral_norm)
        self.mab2 = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, use_layernorm,
            pre_layernorm, is_final_block, use_keras_mha, use_spectral_norm)

    def build(self, input_shape):
        self.inducing_points = self.add_weight(
            shape=(1, self.num_induce, self.embed_dim),
            initializer="glorot_uniform", # xavier_uniform from pytorch implementation
            trainable=True,
            name="Inducing_Points")
        
    def compute_mask(self, inputs, mask):
        return mask

    def call(self, inputs, training=None):
        x = inputs[0]
        mems = inputs[1] 
        
        batch_size = tf.shape(x)[0]
        
        i = tf.tile(self.inducing_points, (batch_size, 1, 1))       
                  
        if mems is not None and mems.shape.ndims > 1:
            concat = tf.concat([mems, x], 1)
        else:
            concat = x
            
        h = self.mab1((i, concat), training=training)
        return self.mab2((x, h), training=training)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.mab2.num_heads,
            "num_induce": self.num_induce,
            "ff_dim": self.mab2.ff_dim,
            "ff_activation": self.mab2.ff_activation,
            "use_layernorm": self.mab2.use_layernorm,
            "pre_layernorm": self.mab2.pre_layernorm,
            "is_final_block": self.mab2.is_final_block,
            "use_keras_mha": self.mab2.use_keras_mha,
            "use_spectral_norm": self.use_spectral_norm
        })
        return config


@CustomLayer
class ConditionedInducedSetAttentionBlock(InducedSetAttentionBlock):
    """
    The Conditioned Induced Set Attention Block (CISAB) transforms sets by performing
    MHA over the set and predicted anchor points. This method was proposed by the paper:
    Generative Adversarial Set Transformer (GAST) by Stelzner et al., available at:
    https://www.ml.informatik.tu-darmstadt.de/papers/stelzner2020ood_gast.pdf
    """

    def build(self, input_shape):
        condition_shape = input_shape[1][1:]
        self.anchor_predictor = keras.models.Sequential([
            spectral_dense(
                self.num_induce*self.embed_dim,
                use_spectral_norm=self.use_spectral_norm,
                input_shape=condition_shape),
            keras.layers.Reshape((self.num_induce, self.embed_dim))
        ], name="Anchor_Points")

    def call(self, inputs: tuple, training=None):
        x, condition = inputs[0], inputs[1]
        anchor_points = self.anchor_predictor(condition)
        h = self.mab1((anchor_points, x), training=training)
        return self.mab2((x, h), training=training)


# Pooling Methods ----------------------------------------------------------------------------------

@CustomLayer
class PoolingByMultiHeadAttention(keras.layers.Layer):
    def __init__(
        self,
        num_seeds,
        embed_dim,
        num_heads,
        ff_dim=None,
        ff_activation="relu",
        use_layernorm=True,
        pre_layernorm=False,
        is_final_block=False,
        use_keras_mha=True,
        use_spectral_norm=False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_seeds = num_seeds
        self.embed_dim = embed_dim

        self.mab = MultiHeadAttentionBlock(
            embed_dim, num_heads, ff_dim, ff_activation, use_layernorm,
            pre_layernorm, is_final_block, use_keras_mha, use_spectral_norm)

        self.seed_vectors = self.add_weight(
            shape=(1, self.num_seeds, self.embed_dim),
            initializer="random_normal",
            trainable=True,
            name="Seeds")


    def call(self, z, training=None):
        batch_size = tf.shape(z)[0]
        seeds = tf.tile(self.seed_vectors, (batch_size, 1, 1))
        return self.mab((seeds, z), training=training)


    def get_config(self):
        config = super().get_config()
        config.update({
            "num_seeds": self.num_seeds,
            "embed_dim": self.embed_dim,
            "num_heads": self.mab.num_heads,
            "ff_dim": self.mab.ff_dim,
            "ff_activation": self.mab.ff_activation,
            "use_layernorm": self.mab.use_layernorm,
            "pre_layernorm": self.mab.pre_layernorm,
            "is_final_block": self.mab.is_final_block,
            "use_keras_mha": self.mab.use_keras_mha,
            "use_spectral_norm": self.mab.use_spectral_norm
        })
        return config


@CustomLayer
class InducedSetEncoder(PoolingByMultiHeadAttention):
    """
    Same as PMA, except resulting rows are summed together.
    """
    def call(self, x):
        out = super().call(x)
        return tf.reduce_sum(out, axis=1)

# --------------------------------------------------------------------------------------------------

# Alias exports
MAB = MultiHeadAttentionBlock
SAB = SetAttentionBlock
ISAB = InducedSetAttentionBlock
CISAB = ConditionedInducedSetAttentionBlock
PMA = PoolingByMultiHeadAttention
ISE = InducedSetEncoder
