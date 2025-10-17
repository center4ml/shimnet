import numpy as np

from .models import ShimNetWithSCRF, Predictor
from .multiscale import MultiscaleFeatureExtractor

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
    model = ShimNetWithSCRF(**config.model.kwargs)
    if config.get("multiscale_features_as_input", False):
        preprocessor = MultiscaleFeatureExtractor(**config.multiscale_features)
    else:
        preprocessor = None
    predictor = Predictor(model, weights_file, preprocessor)
    return predictor
