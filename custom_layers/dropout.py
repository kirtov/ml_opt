from keras.layers import Layer
import keras.backend as K

class AlphaDropout(Layer):
    """Applies Alpha Dropout to the input.

    Alpha Dropout is a Dropout that keeps mean and variance of inputs
to their original values, in order to ensure the self-normalizing property
    even after this dropout.
    Alpha Dropout fits well to Scaled Exponential Linear Units
    by randomly setting activations to the negative saturation value.

    # Arguments
        rate: float, drop probability (as with `Dropout`).
            The multiplicative noise will have
            standard deviation `sqrt(rate / (1 - rate))`.
        seed: A Python integer to use as random seed.

    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.

    # Output shape
        Same shape as input.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    def __init__(self, rate, noise_shape=None, seed=None, **kwargs):
        super(AlphaDropout, self).__init__(**kwargs)
        self.rate = rate
        self.noise_shape = noise_shape
        self.seed = seed
        self.supports_masking = True

    def _get_noise_shape(self, inputs):
        return self.noise_shape if self.noise_shape else K.shape(inputs)

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            noise_shape = self._get_noise_shape(inputs)

            def dropped_inputs(inputs=inputs, rate=self.rate, seed=self.seed):
                alpha = 1.6732632423543772848170429916717
                scale = 1.0507009873554804934193349852946
                alpha_p = -alpha * scale

                kept_idx = K.greater_equal(K.random_uniform(noise_shape, seed=seed), rate)
                kept_idx = K.cast(kept_idx, K.floatx())

                # q is the `keep` probability
                q = 1 - rate
                # Get affine transformation params
                a = (q + alpha_p**2 * q * (1 - q))**(-0.5)
                b = -a * (alpha_p * (1 - q))

                # Apply mask
                x = inputs * kept_idx + alpha_p * (1 - kept_idx)

                # Do affine transformation
                return a * x + b

            return K.in_train_phase(dropped_inputs, inputs, training=training)
        return inputs
