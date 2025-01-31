import abc
import json
import numpy as np

class BaseAttack(abc.ABC):
    """
    Abstract base class for an Attack module.
    All attacks must implement the `attack` method.
    """

    # def __init__(self):
    #     with open('config.json') as json_file:
    #         self.config = json.load(json_file)

    @abc.abstractmethod
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the attack to `audio`.
        Returns a NumPy array of the attacked audio.
        """
        pass

    @property
    def name(self) -> str:
        """Return a short identifier name for this attacker."""
        return self.__class__.__name__