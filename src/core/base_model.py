import abc
import json
import numpy as np
import os
import inspect


class BaseModel(abc.ABC):
    """
    Abstract base class for a Watermarking model.
    """

    def __init__(self):
        model_file = inspect.getfile(self.__class__)
        model_dir = os.path.dirname(os.path.abspath(model_file))

        self.config_path = os.path.join(model_dir, "config.json")

        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"config.json not found in {self.config_path}")

        with open(self.config_path, "r") as json_file:
            self._config = json.load(json_file)

    @abc.abstractmethod
    def embed(
        self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int
    ) -> np.ndarray:
        """
        Embed a watermark into the audio. Return watermarked audio.
        """
        pass

    @abc.abstractmethod
    def detect(self, audio: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        Detect (extract) the watermark from the audio. Return the detected watermark bits.
        """
        pass

    def generate_watermark(self) -> np.ndarray:
        """
        Generate sample watermark.
        """
        return np.random.randint(
            0, 2, size=self.config["watermark_size"], dtype=np.int32
        )

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def config(self) -> dict:
        """Provides read-only access to the attack configuration."""
        return self._config
