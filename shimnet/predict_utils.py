import numpy as np
from hydra.utils import instantiate

from .models import ShimNetWithSCRF, Predictor

class Defaults:
    SCALE = 16.0
    SUFFIX = "_processed"

# functions
def resample_input_spectrum(input_freqs, input_spectrum, Mhz_per_point):
    """resample input spectrum to match the model's frequency range"""
    freqs = np.arange(input_freqs.min(), input_freqs.max(), Mhz_per_point)
    spectrum = np.interp(freqs, input_freqs, input_spectrum)
    return freqs, spectrum

def resample_output_spectrum(input_freqs, freqs, prediction):
    """resample prediction to match the input spectrum's frequency range"""
    prediction = np.interp(input_freqs, freqs, prediction)
    return prediction

def initialize_predictor(config, weights_file):
    if "_target_" in config.model:
        model = instantiate(config.model)
    else:
        model = ShimNetWithSCRF(**config.model.kwargs)
    predictor = Predictor(model, weights_file)
    return predictor

def get_model_ppm_per_point(config):
    return config.data.get("frq_step", config.metadata.get("frq_step")) / config.metadata.spectrometer_frequency