import abc
import json
import numpy as np
import os

class BaseAttack(abc.ABC):
    """
    Abstract base class for an Attack module.
    All attacks must implement the `attack` method.
    Each attack should have its own `config.json` stored in its respective folder.
    """

    def __init__(self):
        attack_file = os.path.abspath(self.__class__.__module__.replace(".", "/") + ".py")
        attack_dir = os.path.dirname(attack_file)
        self.config_path = os.path.join(attack_dir, "config.json")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config.json not found in {self.config_path}")

        with open(self.config_path, "r") as json_file:
            self._config = json.load(json_file)

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
    
    @property
    def config(self) -> dict:
        """Provides read-only access to the attack configuration."""
        return self._config