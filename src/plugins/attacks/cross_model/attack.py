import random

import numpy as np

from core.base_attack import BaseAttack


class CrossModelAttack(BaseAttack):

    def apply(self, audio: np.ndarray, **kwargs) -> np.ndarray:
        """
        Perform multiple watermarking using the same model repeatedly.

        Args:
            audio (np.ndarray): The input audio signal to watermark.
            **kwargs: Additional parameters for the watermarking process.
                - model (BaseModel): Model currently benchmarked.
                - models (dict): Dictionary of all available models.
                - sampling_rate (int): The sampling rate of the audio signal.

        Returns:
            np.ndarray: The watermarked audio signal.
        """
        models = kwargs.get("models", None)
        model = kwargs.get("model", None)
        if model is None:
            raise ValueError(
                "A model must be specified for cross_model_watermarking."
            )
        model = self.get_different_model(model, models)
        sampling_rate = kwargs.get("sampling_rate", None)

        if models is None or sampling_rate is None:
            raise ValueError(
                "A sampling_rate and models must be specified for cross_model_watermarking."
            )

        return model.embed(
            audio=audio,
            watermark_data=model.generate_watermark(),
            sampling_rate=sampling_rate,
        )
    
    def get_different_model(self, model, models):
        """
        Pick a random model from the plugin manager's models,
        excluding the one that's currently being used.
        """
        current_model_name = model.name
        all_model_names = models.keys()
        filtered_list = [name for name in all_model_names if name != current_model_name]

        if not filtered_list:
            raise ValueError(
                f"No other models are available (all we have is '{current_model_name}')."
            )
        key = random.choice(filtered_list)
        model_cls = models[key]['class']
        return model_cls()