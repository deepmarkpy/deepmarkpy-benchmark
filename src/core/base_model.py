import abc
import json
import numpy as np
import os
import inspect


class BaseModel(abc.ABC):
    """
    Abstract base class for a Watermarking model.

    All watermarking models must implement the `embed` and `detect` methods.
    Each model must have a `config.json` file in its respective directory.
    """

    def __init__(self):
        """
        Initializes the watermarking model by loading its configuration file.

        - Determines the file path of the subclass implementing this base class.
        - Constructs the path to `config.json` in the model's directory.
        - Loads the configuration file, raising an error if it is missing.
        """
        model_file = inspect.getfile(self.__class__)  # Get the file path of the subclass
        model_dir = os.path.dirname(os.path.abspath(model_file))  # Get its directory

        self.config_path = os.path.join(model_dir, "config.json")  # Define path to config.json

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config.json not found in {self.config_path}")  # Ensure config exists

        with open(self.config_path, "r") as json_file:
            self._config = json.load(json_file)  # Load the configuration file

    @abc.abstractmethod
    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """
        Embeds a watermark into the given audio signal.

        Args:
            audio (np.ndarray): The input audio signal.
            watermark_data (np.ndarray): The binary watermark data to be embedded.
            sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The watermarked audio signal.

        This method must be implemented by subclasses.
        """
        pass

    @abc.abstractmethod
    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Detects (extracts) the watermark from the given audio signal.

        Args:
            audio (np.ndarray): The input audio signal containing a possible watermark.
            sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The extracted watermark bits.

        This method must be implemented by subclasses.
        """
        pass

    def generate_watermark(self) -> np.ndarray:
        """
        Generates a sample watermark.

        Returns:
            np.ndarray: A randomly generated binary watermark with a length 
                        specified in the model's configuration (`config.json`).
        """
        return np.random.randint(
            0, 2, size=self.config["watermark_size"], dtype=np.int32
        )

    @property
    def name(self) -> str:
        """
        Returns the name of the watermarking model.

        Returns:
            str: The class name of the model instance.
        """
        return self.__class__.__name__

    @property
    def config(self) -> dict:
        """
        Provides read-only access to the model's configuration.

        Returns:
            dict: The model's configuration loaded from `config.json`.
        """
        return self._config