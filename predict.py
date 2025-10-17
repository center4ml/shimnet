import torch
torch.set_grad_enabled(False)
import numpy as np
import argparse
from pathlib import Path
import sys, os
from omegaconf import OmegaConf

from shimnet.predict_utils import Defaults, resample_input_spectrum, resample_output_spectrum, initialize_predictor


# silent deprecation warnings
# https://github.com/pytorch/pytorch/issues/97207#issuecomment-1494781560
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_files", help="Input files", nargs="+") 
    parser.add_argument("--config", help="config file .yaml")
    parser.add_argument("--weights", help="model weights")
    parser.add_argument("-o", "--output_dir", default=".", help="Output directory")
    parser.add_argument("--input_spectrometer_frequency", default=None, type=float, help="spectrometer frequency in MHz (input sample collection frequency). Empty if the same as in the training data")
    parser.add_argument("--suffix", default=Defaults.SUFFIX, help=f"Output file suffix (default: {Defaults.SUFFIX})")
    parser.add_argument("--intensity_scale", default=Defaults.SCALE, type=float, help=f"Intensity scaling factor (default: {Defaults.SCALE})")
    args = parser.parse_args()
    return args
    
# run
if __name__ == "__main__":
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    config = OmegaConf.load(args.config)
    model_ppm_per_point = config.data.frq_step / config.metadata.spectrometer_frequency
    predictor = initialize_predictor(config, args.weights)

    for input_file in args.input_files:
        print(f"processing {input_file} ...")

        # load data
        input_data = np.loadtxt(input_file)
        input_freqs_input_ppm, input_spectrum = input_data[:,0], input_data[:,1]

        # convert input frequencies to model's frequency - correct for zero filling ad spectrometer frequency
        if args.input_spectrometer_frequency is not None:
            input_freqs_model_ppm = input_freqs_input_ppm * args.input_spectrometer_frequency / config.metadata.spectrometer_frequency
        else:
            input_freqs_model_ppm = input_freqs_input_ppm
        
        freqs, spectrum = resample_input_spectrum(input_freqs_model_ppm, input_spectrum, model_ppm_per_point)
        
        spectrum = torch.tensor(spectrum).float()
        # scale height of the spectrum
        scaling_factor = args.intensity_scale / spectrum.max()
        spectrum *= scaling_factor

        # correct spectrum
        prediction = predictor(spectrum).numpy()

        # rescale height
        prediction /= scaling_factor

        # resample the output to match the input spectrum
        output_prediction = resample_output_spectrum(input_freqs_model_ppm, freqs, prediction)

        # save result
        output_file = output_dir / f"{Path(input_file).stem}{args.suffix}{Path(input_file).suffix}"

        np.savetxt(output_file, np.column_stack((input_freqs_input_ppm, output_prediction)))
        print(f"saved to {output_file}")
