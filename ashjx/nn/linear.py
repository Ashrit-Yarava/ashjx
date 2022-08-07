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

    def __init__(self, key: rnd.PRNGKey, in_size: int, out_size: int, use_bias: bool = True):
        """
        * key: PRNG Key.
        * in_size (int): Input dimensions.
        * out_size (int): Output dimensions.
        * use_bias (bool): Whether to use a bias. Default: True
        """
        super(Module, self).__init__()
        w_key, b_key = rnd.split(key, 2)
        self.w = rnd.normal(w_key, shape=(in_size, out_size))
        self.b = jnp.zeros((out_size,)) if use_bias else rnd.normal(
            b_key, shape=(out_size,))

    def __call__(self, x: jnp.ndarray):
        """
        * x (jnp.ndarray): Input for the linear layer.
        """
        x = x @ self.w + self.b
