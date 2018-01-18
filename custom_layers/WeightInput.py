from keras import backend as K
from keras.engine.topology import Layer
from keras.regularizers import l2, l1, l1_l2
import numpy as np
from keras import regularizers

class WeightInput(Layer):
    def __init__(self, output_dim, input_dim,  W_regularizer=l1_l2(0.01), **kwargs):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W_regularizer = regularizers.get(W_regularizer)
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(WeightInput, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        initial_weight_value = np.ones((input_dim))
        self.W = K.variable(initial_weight_value, name='W')
        self.trainable_weights = [self.W]

    def call(self, x, mask=None):
        return x*self.W
  
