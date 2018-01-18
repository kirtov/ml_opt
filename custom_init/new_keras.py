from keras.initializers import VarianceScaling

def selu_normal(seed=None):
    """SELU normal initializer.

    It draws samples from a truncated normal distribution centered on 0
    with `stddev = sqrt(1 / fan_in)`
    where `fan_in` is the number of input units in the weight tensor.

    # Arguments
        seed: A Python integer. Used to seed the random generator.

    # Returns
        An initializer.

    # References
        - [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
    """
    return VarianceScaling(scale=1.,
                           mode='fan_in',
                           distribution='normal',
                           seed=seed)
