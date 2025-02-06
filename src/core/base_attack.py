import abc
import json
import numpy as np
import os
import inspect


class BaseAttack(abc.ABC):
    """
    Abstract base class for an Attack module.
    
    All attacks must implement the `apply` method.
    Each attack should have its own `config.json` stored in its respective folder.
    """

    def __init__(self):
        """
        Initializes the attack by loading its configuration file.
        
        - Determines the file path of the subclass implementing this base class.
        - Constructs the path to `config.json` in the attack's directory.
        - Loads the configuration if the file exists, otherwise sets `_config` to None.
        """
        model_file = inspect.getfile(self.__class__)  # Get the file path of the subclass
        model_dir = os.path.dirname(os.path.abspath(model_file))  # Get its directory

        self.config_path = os.path.join(model_dir, "config.json")  # Define path to config.json

        if not os.path.exists(self.config_path):
            print(f"config.json not found in {self.config_path}")  # Warn if config is missing
            self._config = None  # No config available
        else:
            with open(self.config_path, "r") as json_file:
                self._config = json.load(json_file)  # Load the config file

    @abc.abstractmethod
    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Applies the attack to the given `audio` signal.

        Args:
            audio (np.ndarray): The input audio signal.
            **kwargs: Additional parameters that specific attacks may require.

        Returns:
            np.ndarray: The attacked (modified) audio signal.

        This method must be implemented by all subclasses.
        """
        pass

    @property
    def name(self) -> str:
        """
        Returns a short identifier name for this attack.

        Returns:
            str: The class name of the attack instance.
        """
        return self.__class__.__name__

    @property
    def config(self) -> dict:
        """
        Provides read-only access to the attack configuration.

        Returns:
            dict: The attack's configuration loaded from `config.json`,
                  or None if the file does not exist.
        """
        return self._config