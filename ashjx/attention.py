import jax
import jax.numpy as jnp
from ashjx.module import Module
from ashjx.linear import Linear
from ashjx.params import param, sparam
from typing import Optional, Tuple


class MultiHeadAttention(Module):
    """
    A Multi Head Attention layer. Introduced first in the paper "Attention is all you need".
    Code inspired from the equinox library.
    """

    query: Linear
    key: Linear
    value: Linear
    output: Linear

    num_heads: int = sparam()
    query_size: int = sparam()
    key_size: int = sparam()
    value_size: int = sparam()
    output_size: int = sparam()
    qk_size: int = sparam()
    vo_size: int = sparam()
    use_query_bias: bool = sparam()
    use_key_bias: bool = sparam()
    use_value_bias: bool = sparam()
    use_output_bias: bool = sparam()

    def __init__(self,
            key: jax.random.KeyArray,
            num_heads: int,
            query_size: int,
            key_size: Optional[int] = None,
            value_size: Optional[int] = None,
            output_size: Optional[int] = None,
            query_key_size: Optional[int] = None,
            value_output_size: Optional[int] = None,
            use_query_bias: bool = False,
            use_key_bias: bool = False,
            use_value_bias: bool = False,
            use_output_bias: bool = False,
            ):
        """
        key: The random PRNGKey to use.
        num_heads: Number of attention heads.
        query_size: Number of input channels for the query.
        key_size: Number of input channels for the key.
        value_size: Number of input channels for the value.
        output_size: Number of output channels. Default = query_size
        query_key_size: Number of channels to compare query and key. Default = query_size // num_heads
        value_output_size: Number of channels to compare value and output. Default = query_size // num_heads
        use_*_bias: Whether to use the bias stated for the function.
        """

        super(Module, self).__init__()
        query_key, key_key, value_key, output_key = jax.random.split(key, 4)

        self.num_heads = num_heads
        self.query_size = query_size
        self.key_size = key_size if key_size is not None else query_size
        self.value_size = value_size if value_size is not None else query_size
        self.query_key_size = query_key_size if query_key_size is not None else query_size // num_heads
        self.value_output_size = value_output_size if value_output_size is not None else query_size // num_heads
        self.output_size = output_size if output_size is not None else query_size

        self.query = Linear(query_key, self.query_size, self.num_heads * self.query_key_size, use_bias=use_query_bias)
        self.key = Linear(key_key, self.key_size, self.num_heads * self.query_key_size, use_bias=use_key_bias)
        self.value = Linear(value_key, self.value_size, self.num_heads * self.value_output_size, use_bias=use_value_bias)
        self.output = Linear(output_key, self.num_heads * self.value_output_size, 
                self.output_size, use_bias=use_output_bias)


    def _transform(self, p, x):
        seq_len, _ = x.shape
        p_ = jax.vmap(p)(x)
        return p_.reshape(seq_len, self.num_heads, -1)


    def __call__(self,
            query: jnp.ndarray,
            key: jnp.ndarray,
            value: jnp.ndarray,
            mask: Optional[jnp.ndarray] = None):
        """
        query: The query
        key: The key
        value: The value

        Returns: output (query sequence length, output size)
        """

        query_seq_length, _ = query.shape
        kv_seq_length, _ = key.shape
        kv_seq_length2, _ = value.shape
        if kv_seq_length != kv_seq_length2:
            # query length can be different
            raise ValueError("key and value must both be sequences of equal length.")

        query_heads = self._transform(self.query, query)
        key_heads = self._transform(self.key, key)
        value_heads = self._transform(self.value, value)

        logits = jnp.einsum("shd,Shd->hsS", query_heads, key_heads)
        logits = logits / jnp.sqrt(self.qk_size)
        if mask is not None:
            if mask.shape != logits.shape:
                raise ValueError(
                    f"mask must have shape (num_heads, query_seq_length, "
                    f"kv_seq_length)=({self.num_heads}, {query_seq_length}, "
                    f"{kv_seq_length}). Got {mask.shape}."
                )
            logits = jnp.where(mask, logits, -jnp.inf)

        weights = jax.nn.softmax(logits, axis=-1)
        attn = jnp.einsum("hsS,Shd->shd", weights, value_heads)
        attn = attn.reshape(query_seq_length, -1)

        return jax.vmap(self.output)(attn)


