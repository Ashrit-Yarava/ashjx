import jax
import treeo as to
from abc import abstractmethod


class Module(to.Tree):
    """
    Module
    ---
    A base class to be used for all modules.

    Methods to implement:
    * __init__(key): Initializes the 
        * key: the random PRNG key that should be passed in.
        * args, kwargs: the rest of the parameters.
    * __call__():
        * A call to the module. Performs the associated function.
    """

    @abstractmethod
    def __init__(self, key: jax.random.PRNGKey, *args, **kwargs):
        """
        * key: the random PRNG key that should be passed in.
        * args, kwargs: the rest of the parameters.
        """
        super(self, to.Tree).__init__()
        pass

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        * A call to the module. Performs the associated function.
        """
        pass
