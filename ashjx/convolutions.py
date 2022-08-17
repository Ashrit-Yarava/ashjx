import jax
import jax.numpy as jnp
import jax.random as rnd
import itertools as it
from typing import Sequence
from ashjx.module import Module
from ashjx.params import param, sparam


class Conv(Module):
    """
    Conv
    ---
    Perform a n-dimensional convolution with controllable padding and stride.
    """
    w: jnp.ndarray = param()
    b: jnp.ndarray = param()
    dims: int = sparam()
    kernel_size: Sequence[int] = sparam()
    stride: Sequence[int] = sparam()
    padding: str = sparam()
    dilation: Sequence[int] = sparam()
    use_bias: bool = sparam()


    def __init__(self, 
                 key: rnd.KeyArray,
                 dims: int,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 2,
                 stride: int = 1,
                 padding: str = "same",
                 dilation: int = 1,
                 use_bias: bool = True):
        """
        * dims: The dimension of the convolution (1D, 2D, 3D, ...)
        * key: The random key generator to use for generating initial weights.
        * in_channels: The number of input channels.
        * out_channels: The number of output channels.
        * kernel_size: The size of the convolutional kernel.
        * stride: The stride of the convolution.
        * padding: The padding to be applied.
        * dilation: Dilation of the convolution.
        * use_bias: Whether to use bias at the end.
        """
        wkey, bkey = rnd.split(key, 2)
        self.dims = dims
        self.kernel_size = tuple(it.repeat(kernel_size, dims))
        self.stride = tuple(it.repeat(stride, dims))
        self.dilation = tuple(it.repeat(dilation, dims))
        self.padding = padding
        lim = 1 / jnp.sqrt(in_channels * jnp.prod(kernel_size))
        self.weight = rnd.uniform(wkey, (out_channels, in_channels) + self.kernel_size, minval=-lim, maxval=lim)
        if use_bias:
            self.bias = rnd.uniform(bkey, (out_channels,) + (1,) * dims, minval=-lim, maxval=lim)
        else:
            self.bias = None


    def __call__(self, x: jnp.ndarray):
        """
        Performs the convolution with the padding given in __init__.
        * x: input matrix with shape (in_channels, height, width)
        """
        x = jnp.expand_dims(x, axis=0)
        x = jax.lax.conv_general_dilated(
            lhs=x,
            rhs=self.w,
            window_strides=self.stride,
            padding=self.padding,
            rhs_dilation=self.dilation,
            feature_group_count=1,
        )
        if self.bias:
            x = x + self.bias
        return jnp.squeeze(x, axis=0)
