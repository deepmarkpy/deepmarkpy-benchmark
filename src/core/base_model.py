import abc
import json
import numpy as np


class BaseModel(abc.ABC):
    """
    Abstract base class for a Watermarking model.
    """
    
    @abc.abstractmethod
    def embed(self, audio: np.ndarray, watermark_data: np.ndarray, sampling_rate: int) -> np.ndarray:
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

    @property
    def name(self) -> str:
        return self.__class__.__name__
    
