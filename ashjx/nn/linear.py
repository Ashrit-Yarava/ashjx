import jax.numpy as jnp
import jax.random as rnd
from jax import lax
import treeo as to
from ashjx.nn.module import Module


class Linear(Module):
    """
    Linear
    ---
    A simple linear layer with optional bias.
    """
    w: jnp.ndarray = to.node()
    b: jnp.ndarray = to.node()
    use_bias: bool = to.static()


    def __init__(self, key: jnp.ndarray, in_size: int, out_size: int, use_bias: bool = True):
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
