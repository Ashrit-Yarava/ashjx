import jax
import jax.numpy as jnp
import jax.random as rnd
from ashjx.module import Module
from ashjx.params import param, sparam

class Linear(Module):
    """
    Linear
    ---
    A simple linear layer with optional bias.
    """
    w: jnp.ndarray = param()
    b: jnp.ndarray = param()
    use_bias: bool = sparam()


    def __init__(self, key: jax.random.KeyArray, in_size: int, out_size: int, use_bias: bool = True):
        """
        * key: PRNG Key.
        * in_size (int): Input dimensions.
        * out_size (int): Output dimensions.
        * use_bias (bool): Whether to use a bias. Default: True
        """
        super(Module, self).__init__()
        w_key, b_key = rnd.split(key, 2)
        self.w = rnd.uniform(w_key, shape=(in_size, out_size))
        if use_bias:
            self.b = rnd.uniform(b_key, shape=(out_size,))
        else:
            self.b = jnp.zeros((out_size,))
        self.use_bias = use_bias


    def __call__(self, x: jnp.ndarray):
        """
        * x (jnp.ndarray): Input for the linear layer.
        """
        x = x @ self.w
        if self.use_bias:
            return x + self.b
        else:
            return x
