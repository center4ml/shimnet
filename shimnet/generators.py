from enum import Enum
from copy import deepcopy
from typing import Optional
# from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torchdata
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

def random_uniform(min_value, max_value, generator=None):
    return (min_value + torch.rand(1, generator=generator) * (max_value - min_value)).item()
random_value = random_uniform

def random_loguniform(min_value, max_value, generator=None):
    return (min_value * torch.exp(torch.rand(1, generator=generator) * (torch.log(torch.tensor(max_value)) - torch.log(torch.tensor(min_value))))).item()

def random_uniform_vector(min_value, max_value, size, generator=None):
    return min_value + torch.rand(size, generator=generator) * (max_value - min_value)

def random_loguniform_vector(min_value, max_value, size, generator=None):
    return min_value * torch.exp(torch.rand(size, generator=generator) * (torch.log(torch.tensor(max_value)) - torch.log(torch.tensor(min_value))))

def spectrum_from_peaks_data(peaks_parameters: dict | list, frq_frq:torch.Tensor, relative_frequency=False):

    if isinstance(peaks_parameters, dict):
        peaks_parameters = [peaks_parameters]
    
    spectrum = torch.zeros((1, frq_frq.shape[0]))
    for peak_params in peaks_parameters:
        # extract parameters
        if relative_frequency:
            tff_lin = frq_frq[0] + peak_params["tff_relative"]*(frq_frq[1]-frq_frq[0])
        else:
            tff_lin = peak_params["tff_lin"]
        twf_lin = peak_params["twf_lin"]
        thf_lin = peak_params["thf_lin"]
        trf_lin = peak_params["trf_lin"]

        lwf_lin = twf_lin
        lhf_lin = thf_lin * (1. - trf_lin)
        gwf_lin = twf_lin
        gdf_lin = gwf_lin / torch.tensor(2.).log().mul(2.).sqrt()
        ghf_lin = thf_lin * trf_lin
        # calculate Lorenz peaks contriubutions
        lsf_linfrq = lwf_lin[:, None] ** 2 / (lwf_lin[:, None] ** 2 + (frq_frq - tff_lin[:, None]) ** 2) * lhf_lin[:, None]
        # calculate Gaussian peaks contriubutions
        gsf_linfrq = torch.exp(-(frq_frq - tff_lin[:, None]) ** 2 / gdf_lin[:, None] ** 2 / 2.) * ghf_lin[:, None]
        tsf_linfrq = lsf_linfrq + gsf_linfrq
        # sum peaks contriubutions
        spectrum += tsf_linfrq.sum(0, keepdim = True)
    return spectrum

calculate_theoretical_spectrum = spectrum_from_peaks_data  # Alias for backward compatibility

pascal_triangle = [(1,), (1,1), (1,2,1), (1,3,3,1), (1,4,6,4,1), (1,5,10,10,5,1), (1,6,15,20,15,6,1), (1,7, 21,35,35,21,7,1)]
normalized_pascal_triangle = [torch.tensor(x)/sum(x) for x in pascal_triangle]

def pascal_multiplicity(multiplicity):
    intensities = normalized_pascal_triangle[multiplicity-1]
    n_peaks = len(intensities)
    shifts = torch.arange(n_peaks)-((n_peaks-1)/2)
    return shifts, intensities

def double_multiplicity(multiplicity1, multiplicity2, j1=1, j2=1):
    shifts1, intensities1 = pascal_multiplicity(multiplicity1)
    shifts2, intensities2 = pascal_multiplicity(multiplicity2)

    shifts = (j1*shifts1.reshape(-1,1) + j2*shifts2.reshape(1,-1)).flatten()
    intensities = (intensities1.reshape(-1,1) * intensities2.reshape(1,-1)).flatten()
    return shifts, intensities
    
def generate_multiplet_parameters(multiplicity, tff_lin, thf_lin, twf_lin, trf_lin, j1, j2):
    shifts, intensities = double_multiplicity(multiplicity[0], multiplicity[1], j1, j2)
    n_peaks = len(shifts)

    return {
        "tff_lin": shifts + tff_lin,
        "thf_lin": intensities * thf_lin,
        "twf_lin": torch.full((n_peaks,), twf_lin),
        "trf_lin": torch.full((n_peaks,), trf_lin),
    }

def value_to_index(values, table):
    span = table[-1] - table[0]
    indices = ((values - table[0])/span * (len(table)-1)) #.round().type(torch.int64)
    return indices

def generate_theoretical_spectrum(
    number_of_signals_min, number_of_signals_max,
    spectrum_width_min, spectrum_width_max,
    relative_width_min, relative_width_max, 
    tff_min, tff_max,
    thf_min, thf_max,
    trf_min, trf_max,
    relative_height_min, relative_height_max,
    multiplicity_j1_min, multiplicity_j1_max,
    multiplicity_j2_min, multiplicity_j2_max,
    atom_groups_data,
    frq_frq,
    generator=None
):
    number_of_signals = torch.randint(number_of_signals_min, number_of_signals_max+1, [], generator=generator)
    atom_group_indices = torch.randint(0, len(atom_groups_data), [number_of_signals], generator=generator)
    width_spectrum = random_loguniform(spectrum_width_min, spectrum_width_max, generator=generator)
    height_spectrum = random_loguniform(thf_min, thf_max, generator=generator)
    
    peak_parameters_data = []
    theoretical_spectrum = None
    for atom_group_index in atom_group_indices:
        relative_intensity, multiplicity1, multiplicity2 = atom_groups_data[atom_group_index]
        position = random_value(tff_min, tff_max, generator=generator)
        j1 = random_value(multiplicity_j1_min, multiplicity_j1_max, generator=generator)
        j2 = random_value(multiplicity_j2_min, multiplicity_j2_max, generator=generator)
        width = width_spectrum*random_loguniform(relative_width_min, relative_width_max, generator=generator)
        height = height_spectrum*relative_intensity*random_loguniform(relative_height_min, relative_height_max, generator=generator)
        gaussian_contribution = random_value(trf_min, trf_max, generator=generator)

        peaks_parameters = generate_multiplet_parameters(multiplicity=(multiplicity1, multiplicity2), tff_lin=position, thf_lin=height, twf_lin= width, trf_lin= gaussian_contribution, j1=j1, j2=j2)
        peaks_parameters["tff_relative"] = value_to_index(peaks_parameters["tff_lin"], frq_frq)
        peak_parameters_data.append(peaks_parameters)
        spectrum_contribution = calculate_theoretical_spectrum(peaks_parameters, frq_frq)
        if theoretical_spectrum is None:
            theoretical_spectrum = spectrum_contribution
        else:
            theoretical_spectrum += spectrum_contribution
    return theoretical_spectrum, peak_parameters_data


def theoretical_generator(
    atom_groups_data,
    pixels=2048, frq_step=11160.7142857 / 32768,
    number_of_signals_min=1, number_of_signals_max=8,
    spectrum_width_min=0.2, spectrum_width_max=1,
    relative_width_min=1, relative_width_max=2,
    relative_height_min=1, relative_height_max=1,
    relative_frequency_min=-0.4, relative_frequency_max=0.4,
    thf_min=1/16, thf_max=16,
    trf_min=0, trf_max=1,
    multiplicity_j1_min=0, multiplicity_j1_max=15,
    multiplicity_j2_min=0, multiplicity_j2_max=15,
    ):
    tff_min = relative_frequency_min * pixels * frq_step
    tff_max = relative_frequency_max * pixels * frq_step
    frq_frq = torch.arange(-pixels // 2, pixels // 2) * frq_step
    
    while True:
        yield generate_theoretical_spectrum(
            number_of_signals_min=number_of_signals_min,
            number_of_signals_max=number_of_signals_max,
            spectrum_width_min=spectrum_width_min,
            spectrum_width_max=spectrum_width_max,
            relative_width_min=relative_width_min,
            relative_width_max=relative_width_max,
            relative_height_min=relative_height_min,
            relative_height_max=relative_height_max,
            tff_min=tff_min, tff_max=tff_max,
            thf_min=thf_min, thf_max=thf_max,
            trf_min=trf_min, trf_max=trf_max,
            multiplicity_j1_min=multiplicity_j1_min,
            multiplicity_j1_max=multiplicity_j1_max,
            multiplicity_j2_min=multiplicity_j2_min,
            multiplicity_j2_max=multiplicity_j2_max,
            atom_groups_data=atom_groups_data,
            frq_frq=frq_frq
        )

class ResponseLibrary:
    def __init__(self, response_files, normalize=True):
        self.data = [torch.load(f, map_location='cpu', weights_only=True).flatten(0,-4) for f in response_files]
        if normalize:
            self.data = [data/torch.sum(data, dim=(-1,), keepdim=True) for data in self.data]
        lengths = [len(data) for data in self.data]
        self.start_indices = torch.cumsum(torch.tensor([0] + lengths[:-1]), 0)
        self.total_length = sum(lengths)
    
    def __getitem__(self, idx):
        if idx >= self.total_length:
            raise ValueError(f'index {idx} out of range')
        tensor_index = torch.searchsorted(self.start_indices, idx, right=True) - 1
        return self.data[tensor_index][idx - self.start_indices[tensor_index]]
    
    def __len__(self):
        return self.total_length
    
    @property
    def max_response_length(self):
        return max([data.shape[-1] for data in self.data])

def generator(
    theoretical_generator_params,
    response_function_library,
    response_function_stretch_min=0.5,
    response_function_stretch_max=2.0,
    response_function_noise=0.,
    spectrum_noise_min=0.,
    spectrum_noise_max=1/64,
    include_spectrum_data=False,
    include_peak_mask=False,
    include_response_function=False,
    flip_response_function=False
    
):
    for theoretical_spectrum, theoretical_spectrum_data in theoretical_generator(**theoretical_generator_params):
        # get response function
        response_function = response_function_library[torch.randint(0, len(response_function_library), [1])][0]
        # stretch response function
        padding_size = (response_function.shape[-1] - 1)//2
        padding_size = round(random_loguniform(response_function_stretch_min, response_function_stretch_max)*padding_size) #torch.randint(round(padding_size*response_function_stretch_min), round(paddingSize*response_function_stretch_max), [1]).item()
        response_function = torch.nn.functional.interpolate(response_function, size=2*padding_size+1, mode='linear')        
        response_function /= response_function.sum() # normalize sum of response function to 1
        # add noise to response function
        response_function += torch.randn(response_function.shape) * response_function_noise
        response_function /= response_function.sum() # normalize sum of response function to 1
        if flip_response_function and (torch.rand(1).item() < 0.5):
            response_function = response_function.flip(-1)
        # disturbed spectrum
        disturbed_spectrum = torch.nn.functional.conv1d(theoretical_spectrum, response_function, padding=padding_size)
        # add noise
        noised_spectrum = disturbed_spectrum + torch.randn(disturbed_spectrum.shape) * random_value(spectrum_noise_min, spectrum_noise_max)
        
        out = {
            # 'response_function': response_function,
            'theoretical_spectrum': theoretical_spectrum,
            'disturbed_spectrum': disturbed_spectrum,
            'noised_spectrum': noised_spectrum,
        }
        if include_response_function:
            out['response_function'] = response_function
        if include_spectrum_data:
            out["theoretical_spectrum_data"] = theoretical_spectrum_data
        if include_peak_mask:
            all_peaks_rel = torch.cat([peak_data["tff_relative"] for peak_data in theoretical_spectrum_data])
            peaks_indices = all_peaks_rel.round().type(torch.int64)
            out["peaks_mask"] = torch.scatter(torch.zeros(out["theoretical_spectrum"].shape[1]), 0, peaks_indices, 1.).unsqueeze(0)

        yield out

        
def collate_with_spectrum_data(batch):
    tensor_keys = set(batch[0].keys())
    tensor_keys.remove('theoretical_spectrum_data')
    out = {k: torch.stack([item[k] for item in batch]) for k in tensor_keys}    
    out["theoretical_spectrum_data"] = [item["theoretical_spectrum_data"] for item in batch]
    return out

def get_datapipe(
    response_functions_files,
    atom_groups_data_file=None,
    batch_size=64,
    pixels=2048, frq_step=11160.7142857 / 32768,
    number_of_signals_min=1, number_of_signals_max=8,
    spectrum_width_min=0.2, spectrum_width_max=1,
    relative_width_min=1, relative_width_max=2,
    relative_height_min=1, relative_height_max=1,
    relative_frequency_min=-0.4, relative_frequency_max=0.4,
    thf_min=1/16, thf_max=16,
    trf_min=0, trf_max=1,
    multiplicity_j1_min=0, multiplicity_j1_max=15,
    multiplicity_j2_min=0, multiplicity_j2_max=15,
    response_function_stretch_min=0.5,
    response_function_stretch_max=2.0,
    response_function_noise=0.,
    spectrum_noise_min=0.,
    spectrum_noise_max=1/64,
    include_spectrum_data=False,
    include_peak_mask=False,
    include_response_function=False,
    flip_response_function=False
):
    # singlets
    if atom_groups_data_file is None:
        atom_groups_data = np.ones((1,3), dtype=int)
    else:
        atom_groups_data = np.loadtxt(atom_groups_data_file, usecols=(1,2,3), dtype=int)
    response_function_library = ResponseLibrary(response_functions_files)
    g = generator(
        theoretical_generator_params=dict(
            atom_groups_data=atom_groups_data,
            pixels=pixels, frq_step=frq_step,
            number_of_signals_min=number_of_signals_min, number_of_signals_max=number_of_signals_max,
            spectrum_width_min=spectrum_width_min, spectrum_width_max=spectrum_width_max,
            relative_width_min=relative_width_min, relative_width_max=relative_width_max,
            relative_height_min=relative_height_min, relative_height_max=relative_height_max,
            relative_frequency_min=relative_frequency_min, relative_frequency_max=relative_frequency_max,
            thf_min=thf_min, thf_max=thf_max,
            trf_min=trf_min, trf_max=trf_max,
            multiplicity_j1_min=multiplicity_j1_min, multiplicity_j1_max=multiplicity_j1_max,
            multiplicity_j2_min=multiplicity_j2_min, multiplicity_j2_max=multiplicity_j2_max
        ),
        response_function_library=response_function_library,
        response_function_stretch_min=response_function_stretch_min,
        response_function_stretch_max=response_function_stretch_max,
        response_function_noise=response_function_noise,
        spectrum_noise_min=spectrum_noise_min,
        spectrum_noise_max=spectrum_noise_max,
        include_spectrum_data=include_spectrum_data,
        include_peak_mask=include_peak_mask,
        include_response_function=include_response_function,
        flip_response_function=flip_response_function
    )
    
    pipe = torchdata.datapipes.iter.IterableWrapper(g, deepcopy=False)
    pipe = pipe.batch(batch_size)
    pipe = pipe.collate(collate_fn=collate_with_spectrum_data if include_spectrum_data else None)
    
    return pipe

    # response_functions_files,
    # atom_groups_data_file=None,
    # batch_size=64,
    # pixels=2048, frq_step=11160.7142857 / 32768,
    # number_of_signals_min=1, number_of_signals_max=8,
    # spectrum_width_min=0.2, spectrum_width_max=1,
    # relative_width_min=1, relative_width_max=2,
    # relative_height_min=1, relative_height_max=1,
    # relative_frequency_min=-0.4, relative_frequency_max=0.4,
    # thf_min=1/16, thf_max=16,
    # trf_min=0, trf_max=1,
    # multiplicity_j1_min=0, multiplicity_j1_max=15,
    # multiplicity_j2_min=0, multiplicity_j2_max=15,
    # response_function_stretch_min=0.5,
    # response_function_stretch_max=2.0,
    # response_function_noise=0.,
    # spectrum_noise_min=0.,
    # spectrum_noise_max=1/64,
    # include_spectrum_data=False,
    # include_peak_mask=False,
    # include_response_function=False,
    # flip_response_function=False


class RngGetter:
    def __init__(self, seed=42):
        self.rng = torch.Generator()
        if seed is not None:
            self.rng.manual_seed(seed)
        else:
            self.rng.seed()
    
    def get_rng(self, seed=None):
        # Use provided seed or fall back to instance RNG
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
        else:
            rng = self.rng
        return rng


class PeaksParameterDataGenerator:
    """
    Generates peak parameter data for NMR multiplets.
    
    This class is responsible for generating the parameters that describe individual peaks
    in an NMR spectrum (frequencies, heights, widths, Gaussian/Lorentzian ratio).
    """
    def __init__(self, 
                 tff_min=None, #may be assigned after initialization
                 tff_max=None, #may be assigned after initialization
                 atom_groups_data_file=None,
                 number_of_signals_min=1, 
                 number_of_signals_max=8,
                 relative_frequency_min=-0.4,
                 relative_frequency_max=0.4,
                 spectrum_width_min=0.2, 
                 spectrum_width_max=1,
                 relative_width_min=1, 
                 relative_width_max=2,
                 relative_height_min=1, 
                 relative_height_max=1,
                 thf_min=1/16, 
                 thf_max=16,
                 trf_min=0, 
                 trf_max=1,
                 multiplicity_j1_min=0, 
                 multiplicity_j1_max=15,
                 multiplicity_j2_min=0, 
                 multiplicity_j2_max=15,
                 seed=42
                 ):
        # Read atom_groups_data from file
        if atom_groups_data_file is None:
            self.atom_groups_data = np.ones((1,3), dtype=int)
        else:
            self.atom_groups_data = np.atleast_2d(np.loadtxt(atom_groups_data_file, usecols=(1,2,3), dtype=int))
        
        self.tff_min = tff_min
        self.tff_max = tff_max
        self.number_of_signals_min = number_of_signals_min
        self.number_of_signals_max = number_of_signals_max
        self.relative_frequency_min = relative_frequency_min
        self.relative_frequency_max = relative_frequency_max

        self.spectrum_width_min = spectrum_width_min
        self.spectrum_width_max = spectrum_width_max
        self.relative_width_min = relative_width_min
        self.relative_width_max = relative_width_max
        self.relative_height_min = relative_height_min
        self.relative_height_max = relative_height_max
        self.thf_min = thf_min
        self.thf_max = thf_max
        self.trf_min = trf_min
        self.trf_max = trf_max
        self.multiplicity_j1_min = multiplicity_j1_min
        self.multiplicity_j1_max = multiplicity_j1_max
        self.multiplicity_j2_min = multiplicity_j2_min
        self.multiplicity_j2_max = multiplicity_j2_max
        
        self.rng_getter = RngGetter(seed=seed)
    
    def set_frq_range(self, frq_min, frq_max):
        self.tff_min = frq_min * self.relative_frequency_min
        self.tff_max = frq_max * self.relative_frequency_max

    def __call__(self, seed=None):
        """
        Generate peak parameters data.
        
        Args:
            seed: Optional seed for reproducibility
        
        Returns:
            List of dicts containing peak parameters (without tff_relative)
        """
        if self.tff_min is None or self.tff_max is None:
            raise ValueError("tff_min and tff_max must be set before calling the generator.")
    
        rng = self.rng_getter.get_rng(seed=seed)
        
        number_of_signals = torch.randint(
            self.number_of_signals_min, 
            self.number_of_signals_max + 1, 
            [], 
            generator=rng
        )
        atom_group_indices = torch.randint(
            0, 
            len(self.atom_groups_data), 
            [number_of_signals], 
            generator=rng
        )
        width_spectrum = random_loguniform(
            self.spectrum_width_min, 
            self.spectrum_width_max, 
            generator=rng
        )
        height_spectrum = random_loguniform(
            self.thf_min, 
            self.thf_max, 
            generator=rng
        )
        
        peaks_parameters_data = []
        for atom_group_index in atom_group_indices:
            relative_intensity, multiplicity1, multiplicity2 = self.atom_groups_data[atom_group_index]
            position = random_value(self.tff_min, self.tff_max, generator=rng)
            j1 = random_value(self.multiplicity_j1_min, self.multiplicity_j1_max, generator=rng)
            j2 = random_value(self.multiplicity_j2_min, self.multiplicity_j2_max, generator=rng)
            width = width_spectrum * random_loguniform(
                self.relative_width_min, 
                self.relative_width_max, 
                generator=rng
            )
            height = height_spectrum * relative_intensity * random_loguniform(
                self.relative_height_min, 
                self.relative_height_max, 
                generator=rng
            )
            gaussian_contribution = random_value(self.trf_min, self.trf_max, generator=rng)
            
            peak_parameters = generate_multiplet_parameters(
                multiplicity=(multiplicity1, multiplicity2), 
                tff_lin=position, 
                thf_lin=height, 
                twf_lin=width, 
                trf_lin=gaussian_contribution, 
                j1=j1, 
                j2=j2
            )
            peaks_parameters_data.append(peak_parameters)

        return peaks_parameters_data

class TheoreticalMultipletSpectraGenerator:
    """
    Generates theoretical NMR multiplet spectra.
    
    This class combines peak parameter generation with spectrum calculation.
    It can accept either a PeaksParameterDataGenerator instance or parameters to create one.
    """
    def __init__(self, 
                 peaks_parameter_generator,
                 pixels=2048, 
                 frq_step=11160.7142857 / 32768,
                 relative_frequency_min=-0.4, 
                 relative_frequency_max=0.4,
                 frequency_min=None, #if None, the 0 will be in the center of spectrum
                 frequency_max=None,
                 include_tff_relative=False,
                 seed=42
                 ):

        # Spectrum-level parameters
        self.pixels = pixels
        self.frq_step = frq_step
        self.relative_frequency_min = relative_frequency_min
        self.relative_frequency_max = relative_frequency_max
        self.include_tff_relative = include_tff_relative
        # Frequency axis
        self.frq_frq, frq_min, frq_max = self._frequency_axis_from_parameters(frq_step, pixels, frequency_min, frequency_max)
        
        self.peaks_parameter_generator = peaks_parameter_generator
        self.peaks_parameter_generator.set_frq_range(frq_min, frq_max)

        # self.rng_getter = RngGetter(seed=seed) # self.rng_getter.get_rng(seed=seed) to get random generator

    def _frequency_axis_from_parameters(self, frq_step, pixels, frequency_min, frequency_max):
        """frq_step is never None, pixels, frequency_min or frequency_max can be None
        """
        # Option 1: from pixels and frq_step
        if pixels is not None:
            assert (frequency_min is None) or (frequency_max is None)
            if (frequency_min is None) and (frequency_max is None): # if both are None, center at 0
                frequency_min = -(pixels // 2) * frq_step
            elif frequency_min is None: # frequency_max is not None, use it to calculate frequency_min
                frequency_min = frequency_max - pixels * frq_step 
            frq_frq = torch.arange(0, pixels) * frq_step + frequency_min
        # Option 2: from frequency_min and frequency_max
        elif (frequency_min is not None) and (frequency_max is not None):
            pixels = round((frequency_max - frequency_min) / frq_step)
            frq_frq = torch.arange(0, pixels) * frq_step + frequency_min
        else:
            raise ValueError("Insufficient parameters to determine frequency axis.")
        
        return frq_frq, frq_frq[0], frq_frq[-1]

        
    def __call__(self, seed=None):
        """
        Generate a theoretical spectrum.
        
        Args:
            seed: Optional seed for reproducibility
        
        Returns:
            Tuple of (spectrum, dict with spectrum_data and frq_frq)
        """
        # Generate peak parameters (peaks_parameter_generator has its own RngGetter)
        peaks_parameters_data = self.peaks_parameter_generator(seed=seed)
        
        # Add tff_relative if requested
        if self.include_tff_relative:
            for peak_params in peaks_parameters_data:
                peak_params["tff_relative"] = value_to_index(peak_params["tff_lin"], self.frq_frq)
        
        # Create spectrum from peaks
        spectrum = spectrum_from_peaks_data(peaks_parameters_data, self.frq_frq)

        return spectrum, {"spectrum_data": peaks_parameters_data, "frq_frq": self.frq_frq}


class PeaksParametersNames(Enum):
    """Enum for standardized peak parameter names."""
    position_hz ="tff_lin"
    height = "thf_lin"
    width_hz = "twf_lin"
    gaussian_fraction = "trf_lin"

    @classmethod
    def keys(cls):
        return [member.value for member in cls]
    
    @classmethod
    def values(cls):
        return [member.name for member in cls]

class PeaksParametersParser:
    """class to convert peaks parameters from `{"width_hz": [...], "height": ..., ...}` format to `{"twf_lin": torch.tensor([...]), "thf_lin": ..., ...}` format."""
    def __init__(self,
        alias_position_hz = None,
        alias_height = None,
        alias_width_hz = None,
        alias_gaussian_fraction = None,
        default_position_hz = None,
        default_height = None,
        default_width_hz = None,
        default_gaussian_fraction = 0.,
        ):
        self.alias_position_hz = alias_position_hz if alias_position_hz is not None else "position_hz"
        self.alias_height = alias_height if alias_height is not None else "height"
        self.alias_width_hz = alias_width_hz if alias_width_hz is not None else "width_hz"
        self.alias_gaussian_fraction = alias_gaussian_fraction if alias_gaussian_fraction is not None else "gaussian_fraction"
        self.default_position_hz = default_position_hz
        self.default_height = default_height
        self.default_width_hz = default_width_hz
        self.default_gaussian_fraction = default_gaussian_fraction

    def transform_single_peak(self, peak: dict) -> dict:
        parsed_peak = {
            PeaksParametersNames.position_hz.value: peak.get(self.alias_position_hz, self.default_position_hz),
            PeaksParametersNames.height.value: peak.get(self.alias_height, self.default_height),
            PeaksParametersNames.width_hz.value: peak.get(self.alias_width_hz, self.default_width_hz),
            PeaksParametersNames.gaussian_fraction.value: peak.get(self.alias_gaussian_fraction, self.default_gaussian_fraction),
        }
        # Validate and convert other peak parameters
        for k, v in parsed_peak.items():
            if v is None:
                raise ValueError(f"Peak parameter '{k}' is None.")
            parsed_peak[k] = torch.atleast_1d(v.float() if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float32))
        return parsed_peak

    def transform(self, spectrum_peaks: list[dict]) -> list[dict]:
        parsed_peaks = []
        for peak in spectrum_peaks:
            parsed_peaks.append(self.transform_single_peak(peak))
        return parsed_peaks

def csv_file_to_multiplets_dict(file_path: str) -> list[dict]:
    peaks_data = pd.read_csv(file_path)
    multiplets = {k: v.drop(columns="multiplet_name").to_dict(orient='list') for k, v in peaks_data.groupby("multiplet_name")}
    return multiplets

def combine_multiplets(multiplets_list: list[dict]) -> dict:
    composed_multiplets = {}
    for multiplets in multiplets_list:
        for k, v in multiplets.items():
            if not k in composed_multiplets:
                composed_multiplets[k] = v
            else:
                composed_multiplets[k].extend(v)
    return composed_multiplets

class MultipletsLibrary:
    def __init__(self, csv_files_paths: list[str], peak_data_parser: PeaksParametersParser = None, return_name=False):
        self.csv_files_paths = csv_files_paths
        self.multiplets_data = {}
        self.peak_data_parser = peak_data_parser
        for file_path in csv_files_paths:
            self.multiplets_data.update(self._get_multiplet_data_from_file(file_path))
        
        self.names = sorted(self.multiplets_data.keys())
        self.return_name = return_name

    def _get_multiplet_data_from_file(self, file_path: str) -> dict:
        multiplets = csv_file_to_multiplets_dict(file_path) # dict[dict]
        multiplets_out = {}
        for k, v in multiplets.items():
            multiplets_out[f"{file_path}/{k}"] = self.peak_data_parser.transform([v])[0] if self.peak_data_parser else v
        return multiplets_out

    def get_by_name(self, name: str) -> dict:
        return self.multiplets_data.get(name, None)
    
    def __getitem__(self, idx: int) -> dict:
        name = self.names[idx]
        multiplet_data = deepcopy(self.multiplets_data[name])
        if self.return_name:
            return name, multiplet_data
        return multiplet_data
    
    def __len__(self):
        return len(self.multiplets_data)

class SectraLibrary(MultipletsLibrary):
    def _get_multiplet_data_from_file(self, file_path: str) -> dict:
        multiplets = csv_file_to_multiplets_dict(file_path) # dict[dict]
        combined_multiplet = combine_multiplets(multiplets.values()) # dict
        return {f"{file_path}": self.peak_data_parser.transform([combined_multiplet])[0]}

class MultipletDataFromMultipletsLibrary:
    def __init__(self,
        multiplets_library,
        tff_min=None, #may be assigned after initialization if the original peak positions are not used
        tff_max=None, #may be assigned after initialization if the original peak positions are not used
        use_original_peak_position=True,
        number_of_signals_min=None, 
        number_of_signals_max=None,
        relative_frequency_min=None,
        relative_frequency_max=None,
        spectrum_width_factor_min=1, 
        spectrum_width_factor_max=1,
        multiplet_width_factor_min=1, 
        multiplet_width_factor_max=1,
        multiplet_width_additive_min=0, 
        multiplet_width_additive_max=0,
        spectrum_height_factor_min=1, 
        spectrum_height_factor_max=1,
        multiplet_height_factor_min=1, 
        multiplet_height_factor_max=1,
        multiplet_height_additive_min=0, 
        multiplet_height_additive_max=0,
        position_shift_min=0, 
        position_shift_max=0,
        gaussian_fraction_change_min=None, 
        gaussian_fraction_change_max=None,
        gaussian_fraction_change_additive_min=0.,
        gaussian_fraction_change_additive_max=0.,
        seed=42
        ):

        if (number_of_signals_min is None) != (number_of_signals_max is None):
            raise ValueError("Both number_of_signals_min and number_of_signals_max should be provided or both should be None.")

        self.multiplets_library = multiplets_library
        self.rng_getter = RngGetter(seed=seed)
        self.tff_min = tff_min
        self.tff_max = tff_max
        self.relative_frequency_min = relative_frequency_min
        self.relative_frequency_max = relative_frequency_max
        self.use_original_peak_position = use_original_peak_position
        self.number_of_signals_min = number_of_signals_min
        self.number_of_signals_max = number_of_signals_max
        self.spectrum_width_factor_min = spectrum_width_factor_min
        self.spectrum_width_factor_max = spectrum_width_factor_max
        self.multiplet_width_factor_min = multiplet_width_factor_min
        self.multiplet_width_factor_max = multiplet_width_factor_max
        self.multiplet_width_additive_min = multiplet_width_additive_min
        self.multiplet_width_additive_max = multiplet_width_additive_max
        self.spectrum_height_factor_min = spectrum_height_factor_min
        self.spectrum_height_factor_max = spectrum_height_factor_max
        self.multiplet_height_factor_min = multiplet_height_factor_min
        self.multiplet_height_factor_max = multiplet_height_factor_max
        self.multiplet_height_additive_min = multiplet_height_additive_min
        self.multiplet_height_additive_max = multiplet_height_additive_max
        self.position_shift_min = position_shift_min
        self.position_shift_max = position_shift_max
        self.gaussian_fraction_change_min = gaussian_fraction_change_min
        self.gaussian_fraction_change_max = gaussian_fraction_change_max
        self.gaussian_fraction_change_additive_min = gaussian_fraction_change_additive_min
        self.gaussian_fraction_change_additive_max = gaussian_fraction_change_additive_max

    def set_frq_range(self, frq_min, frq_max):
        self.tff_min = frq_min * self.relative_frequency_min
        self.tff_max = frq_max * self.relative_frequency_max


    def __call__(self, seed=None):
        if (not self.use_original_peak_position) and (self.tff_min is None or self.tff_max is None):
            raise ValueError("for use_original_peak_position=False, tff_min and tff_max must be set before calling the generator.")

        rng = self.rng_getter.get_rng(seed=seed)

        # select number of signals and their indices
        if self.number_of_signals_min is None:
            number_of_signals = len(self.multiplets_library)
            multiplets_indices = list(range(len(self.multiplets_library)))
        else:
            number_of_signals = torch.randint(
                self.number_of_signals_min, 
                self.number_of_signals_max + 1, 
                [], 
                generator=rng
            )
        
            multiplets_indices = torch.randint(
                0, 
                len(self.multiplets_library), 
                [number_of_signals], 
                generator=rng
            )
        
        # spectrum width and height factors
        spectrum_width_factor = random_loguniform(
            self.spectrum_width_factor_min, 
            self.spectrum_width_factor_max, 
            generator=rng
        )

        spectrum_height_factor = random_loguniform(
            self.spectrum_height_factor_min, 
            self.spectrum_height_factor_max, 
            generator=rng
        )

        # get and modify peaks parameters data
        peaks_parameters_data = [self.multiplets_library[idx] for idx in multiplets_indices]
        for peak_parameters in peaks_parameters_data:
            
            # position
            if not self.use_original_peak_position:
                new_position_center = random_value(self.tff_min, self.tff_max, generator=rng)
                peak_parameters["tff_lin"] += new_position_center - torch.mean(peak_parameters["tff_lin"])
            else:
                position_shift = random_value(self.position_shift_min, self.position_shift_max, generator=rng)
                peak_parameters["tff_lin"] += position_shift

            # width
            multiplet_width_factor = random_loguniform(
                self.multiplet_width_factor_min, 
                self.multiplet_width_factor_max, 
                generator=rng
            )
            multiplet_width_additive = random_uniform(
                self.multiplet_width_additive_min, 
                self.multiplet_width_additive_max, 
                generator=rng
            )
            peak_parameters["twf_lin"] = peak_parameters["twf_lin"] * spectrum_width_factor * multiplet_width_factor + multiplet_width_additive

            # height
            multiplet_height_factor = random_loguniform(
                self.multiplet_height_factor_min, 
                self.multiplet_height_factor_max, 
                generator=rng
            )
            multiplet_height_additive = random_uniform(
                self.multiplet_height_additive_min, 
                self.multiplet_height_additive_max, 
                generator=rng
            )
            peak_parameters["thf_lin"] = peak_parameters["thf_lin"] * spectrum_height_factor * multiplet_height_factor + multiplet_height_additive

            # gaussian contribution
            if self.gaussian_fraction_change_min is not None:
                gaussian_contribution_shift = random_value(self.gaussian_fraction_change_min, self.gaussian_fraction_change_max, generator=rng)
                gaussian_contribution_additive = random_value(self.gaussian_fraction_change_additive_min, self.gaussian_fraction_change_additive_max, generator=rng)
                gaussian_contribution_shift += gaussian_contribution_additive
                peak_parameters["trf_lin"] = torch.clip(peak_parameters["trf_lin"] + gaussian_contribution_shift, 0., 1.)

        return peaks_parameters_data


class ResponseGenerator:
    def __init__(self, response_function_library, response_function_stretch_min=1., response_function_stretch_max=1., pad_to=None,
                 response_function_noise=0.0, flip_response_function=False, seed=42):
        self.response_function_library = response_function_library
        self.response_function_stretch_min = response_function_stretch_min
        self.response_function_stretch_max = response_function_stretch_max
        self.pad_to = pad_to
        self.response_function_noise = response_function_noise
        self.flip_response_function = flip_response_function
        self.rng_getter = RngGetter(seed=seed) # self.rng_getter.get_rng(seed=seed) to get random generator 

    def __call__(self, seed=None):
        rng = self.rng_getter.get_rng(seed=seed)
        
        response_function = self.response_function_library[torch.randint(0, len(self.response_function_library), [1], generator=rng)][0]
        padding_size = (response_function.shape[-1] - 1)//2
        padding_size = round(random_loguniform(self.response_function_stretch_min, self.response_function_stretch_max, generator=rng)*padding_size)
        response_function = torch.nn.functional.interpolate(response_function, size=2*padding_size+1, mode='linear')
        response_function /= response_function.sum()
        response_function += torch.randn(response_function.shape, generator=rng) * self.response_function_noise
        response_function /= response_function.sum()
        if self.flip_response_function and (torch.rand(1, generator=rng).item() < 0.5):
            response_function = response_function.flip(-1)
        if self.pad_to is not None:
            pad_size_left = (self.pad_to - response_function.shape[-1]) // 2
            pad_size_right = self.pad_to - response_function.shape[-1] - pad_size_left
            response_function = torch.nn.functional.pad(response_function, (pad_size_left, pad_size_right))
        return response_function

class NoiseGenerator:
    def __init__(self, spectrum_noise_min=0., spectrum_noise_max=1/64, seed=42):
        self.spectrum_noise_min = spectrum_noise_min
        self.spectrum_noise_max = spectrum_noise_max
        self.rng_getter = RngGetter(seed=seed) # self.rng_getter.get_rng(seed=seed) to get random generator 

    def __call__(self, disturbed_spectrum, seed=None):
        rng = self.rng_getter.get_rng(seed=seed)        
        return disturbed_spectrum + torch.randn(disturbed_spectrum.shape, generator=rng) * random_value(self.spectrum_noise_min, self.spectrum_noise_max, generator=rng)

class BaseGenerator(ABC):
    """
    Single-threaded base generator.
    
    For this workload, single-threaded execution is typically faster because:
    - Thread creation/synchronization overhead > computation time
    - Python GIL contention during object creation
    - Memory allocator contention when multiple threads allocate tensors
    - CPU cache thrashing across cores
    - Small per-thread workload doesn't amortize thread overhead
    """
    def __init__(self, batch_size=64, seed=None):
        self.batch_size = batch_size
        self.seed = seed
    
    def set_seed(self, seed):
        self.seed = seed

    @abstractmethod
    def _generate_element(self, seed):
        pass

    def __iter__(self):
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)
        else:
            rng.seed()

        while True:
            batch = []
            # Generate unique seeds for each element in the batch
            if self.seed is not None:
                element_seeds = [torch.randint(0, 2**31, (1,), generator=rng).item() for _ in range(self.batch_size)]
            else:
                element_seeds = [None] * self.batch_size
            
            # Single-threaded sequential generation
            for i in range(self.batch_size):
                batch.append(self._generate_element(element_seeds[i]))
            
            yield self.collate_fn(batch)

    @abstractmethod
    def collate_fn(self, batch):
        pass


class BaseGeneratorMultithread(ABC):
    """
    Multithreaded base generator (backup option).
        
    Use only if profiling shows benefit for your specific use case
    (e.g., very large/slow generation functions, I/O-bound operations).
    """
    def __init__(self, batch_size=64, num_workers=4, seed=None, ordered_batch=False):
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.ordered_batch = ordered_batch

    def set_seed(self, seed):
        self.seed = seed

    def set_ordered_batch(self, ordered_batch):
        self.ordered_batch = ordered_batch

    @abstractmethod
    def _generate_element(self, seed):
        pass

    def __iter__(self):
        rng = torch.Generator()
        if self.seed is not None:
            rng.manual_seed(self.seed)
        else:
            rng.seed()

        while True:
            batch = []
            # Generate unique seeds for each element in the batch
            if self.seed is not None:
                element_seeds = [torch.randint(0, 2**31, (1,), generator=rng).item() for _ in range(self.batch_size)]
            else:
                element_seeds = [None] * self.batch_size
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [executor.submit(self._generate_element, element_seeds[i]) for i in range(self.batch_size)]

                if self.ordered_batch:
                    # Maintain order: iterate futures in submission order
                    for f in futures:
                        batch.append(f.result())
                else:
                    # Faster: process as completed (order may vary)
                    for f in as_completed(futures):
                        batch.append(f.result())
            
            yield self.collate_fn(batch)

    @abstractmethod
    def collate_fn(self, batch):
        pass

class Generator(BaseGenerator):
    def __init__(self, clean_spectra_generator, response_generator, noise_generator, batch_size=64,
                 include_spectrum_data=False, include_peak_mask=False, include_response_function=False, input_normalization_height=None, seed=None):
        super().__init__(batch_size=batch_size, seed=seed)
        self.clean_spectra_generator = clean_spectra_generator
        self.response_generator = response_generator
        self.noise_generator = noise_generator
        self.include_spectrum_data = include_spectrum_data
        self.include_peak_mask = include_peak_mask
        self.include_response_function = include_response_function
        self.input_normalization_height = input_normalization_height

    def _generate_element(self, seed):
        # Generate different seeds for each generator from the provided seed
        if seed is not None:
            rng = torch.Generator()
            rng.manual_seed(seed)
            clean_seed = torch.randint(0, 2**31, (1,), generator=rng).item()
            response_seed = torch.randint(0, 2**31, (1,), generator=rng).item()
            noise_seed = torch.randint(0, 2**31, (1,), generator=rng).item()
        else:
            clean_seed = None
            response_seed = None
            noise_seed = None
        
        clean_spectrum, extra_clean_data = self.clean_spectra_generator(seed=clean_seed)
        response_function = self.response_generator(seed=response_seed)
        padding_size = (response_function.shape[-1] - 1)//2
        disturbed_spectrum = torch.nn.functional.conv1d(clean_spectrum, response_function, padding=padding_size)

        if self.input_normalization_height is not None:
            max_val = torch.max(disturbed_spectrum)
            clean_spectrum = clean_spectrum / max_val * self.input_normalization_height
            disturbed_spectrum = disturbed_spectrum / max_val * self.input_normalization_height

        # noise after normalization to better control noise level
        noised_spectrum = self.noise_generator(disturbed_spectrum, seed=noise_seed)

        out = {
            'theoretical_spectrum': clean_spectrum,
            'disturbed_spectrum': disturbed_spectrum,
            'noised_spectrum': noised_spectrum,
        }
        if self.include_spectrum_data:
            out['theoretical_spectrum_data'] = extra_clean_data['spectrum_data']
            out['frq_frq'] = extra_clean_data['frq_frq']
        if self.include_peak_mask and extra_clean_data is not None:
            all_peaks_rel = torch.cat([peak_data["tff_relative"] for peak_data in extra_clean_data['spectrum_data']])
            peaks_indices = all_peaks_rel.round().type(torch.int64)
            out["peaks_mask"] = torch.scatter(torch.zeros(out["theoretical_spectrum"].shape[1]), 0, peaks_indices, 1.).unsqueeze(0)
        if self.include_response_function:
            out['response_function'] = response_function
        return out

    def collate_fn(self, batch):
        tensor_keys = set(batch[0].keys())
        for k in ['theoretical_spectrum_data', 'frq_frq']:
            tensor_keys.discard(k)
        out = {k: torch.stack([item[k] for item in batch]) for k in tensor_keys}
        if 'theoretical_spectrum_data' in batch[0]:
            out['theoretical_spectrum_data'] = [item['theoretical_spectrum_data'] for item in batch]
        if 'frq_frq' in batch[0]:
            out['frq_frq'] = [item['frq_frq'] for item in batch]
        return out

class PeaksParametersFromSinglets:
    def __init__(self,
        singlets_files: list[pd.DataFrame],
        number_of_signals_min: int = 5,
        number_of_signals_max: int = 20,
        use_original_position: bool = True,
        position_hz_min: Optional[float] = None,
        position_hz_max: Optional[float] = None,
        position_hz_change_min: float = 0.0,
        position_hz_change_max: float = 0.0,
        relative_frequency_min: float = -0.4, # used only if position_hz_min/max are None
        relative_frequency_max: float = 0.4,
        use_original_width: bool = True,
        width_hz_min: float = 0.2,
        width_hz_max: float = 2.0,
        width_factor_min: float = 1.0,
        width_factor_max: float = 1.0,
        width_hz_change_min: float = 0.0,
        width_hz_change_max: float = 0.0,
        use_original_height: bool = True,
        height_min: float = 0.1,
        height_max: float = 10.0,
        height_factor_min: float = 1.0,
        height_factor_max: float = 1.0,
        height_change_min: float = 0.0,
        height_change_max: float = 0.0,
        use_original_gaussian_fraction: bool = True,
        gaussian_fraction_min: float = 0.0,
        gaussian_fraction_max: float = 1.0,
        gaussian_fraction_change_min: float = 0.0,
        gaussian_fraction_change_max: float = 0.0,
        seed=42
    ):
        self.peaks_rows = pd.concat([pd.read_csv(f) for f in singlets_files], ignore_index=True)
        
        # number of signals
        self.number_of_signals_min = number_of_signals_min
        self.number_of_signals_max = number_of_signals_max
        # position
        self.use_original_position = use_original_position
        self.position_hz_min = position_hz_min
        self.position_hz_max = position_hz_max
        self.position_hz_change_min = position_hz_change_min
        self.position_hz_change_max = position_hz_change_max
        self.relative_frequency_min = relative_frequency_min
        self.relative_frequency_max = relative_frequency_max
        # width
        self.use_original_width = use_original_width
        self.width_hz_min = width_hz_min
        self.width_hz_max = width_hz_max
        self.width_factor_min = width_factor_min
        self.width_factor_max = width_factor_max
        self.width_hz_change_min = width_hz_change_min
        self.width_hz_change_max = width_hz_change_max
        # height
        self.use_original_height = use_original_height
        self.height_min = height_min
        self.height_max = height_max
        self.height_factor_min = height_factor_min
        self.height_factor_max = height_factor_max
        self.height_change_min = height_change_min
        self.height_change_max = height_change_max
        # gaussian fraction
        self.use_original_gaussian_fraction = use_original_gaussian_fraction
        self.gaussian_fraction_min = gaussian_fraction_min
        self.gaussian_fraction_max = gaussian_fraction_max
        self.gaussian_fraction_change_min = gaussian_fraction_change_min
        self.gaussian_fraction_change_max = gaussian_fraction_change_max

        self.rng_getter = RngGetter(seed=seed)

    def set_frq_range(self, frq_min, frq_max):
        self.position_hz_min = frq_min * self.relative_frequency_min
        self.position_hz_max = frq_max * self.relative_frequency_max

    def __call__(self, seed=None) -> list[dict]:
        rng = self.rng_getter.get_rng(seed=seed)

        number_of_signals = torch.randint(
            low=self.number_of_signals_min,
            high=min(self.number_of_signals_max, len(self.peaks_rows) + 1),
            size=[],
            generator=rng
        )
        selected_peaks = self.peaks_rows.sample(n=number_of_signals.item(), random_state=seed)

        multiplet_data = {}
        # position
        if self.use_original_position:
            multiplet_data[PeaksParametersNames.position_hz.value] = torch.tensor(selected_peaks["position_hz"].values, dtype=torch.float32) + random_uniform_vector(self.position_hz_change_min, self.position_hz_change_max, size=len(selected_peaks))
        else:
            multiplet_data[PeaksParametersNames.position_hz.value] = random_uniform_vector(self.position_hz_min, self.position_hz_max, size=len(selected_peaks))
        # width
        if self.use_original_width:
            multiplet_data[PeaksParametersNames.width_hz.value] = torch.tensor(selected_peaks["width_hz"].values, dtype=torch.float32) * random_uniform_vector(self.width_factor_min, self.width_factor_max, size=len(selected_peaks)) + random_uniform_vector(self.width_hz_change_min, self.width_hz_change_max, size=len(selected_peaks))
        else:
            multiplet_data[PeaksParametersNames.width_hz.value] = random_loguniform_vector(self.width_hz_min, self.width_hz_max, size=len(selected_peaks))
        # height
        if self.use_original_height:
            multiplet_data[PeaksParametersNames.height.value] = torch.tensor(selected_peaks["height"].values, dtype=torch.float32) * random_uniform_vector(self.height_factor_min, self.height_factor_max, size=len(selected_peaks)) + random_uniform_vector(self.height_change_min, self.height_change_max, size=len(selected_peaks))
        else:
            multiplet_data[PeaksParametersNames.height.value] = random_loguniform_vector(self.height_min, self.height_max, size=len(selected_peaks))
        # gaussian fraction
        if self.use_original_gaussian_fraction:
            multiplet_data[PeaksParametersNames.gaussian_fraction.value] = torch.clamp(torch.tensor(selected_peaks["gaussian_fraction"].values, dtype=torch.float32) + random_uniform_vector(self.gaussian_fraction_change_min, self.gaussian_fraction_change_max, size=len(selected_peaks)), 0.0, 1.0)
        else:
            multiplet_data[PeaksParametersNames.gaussian_fraction.value] = random_uniform_vector(self.gaussian_fraction_min, self.gaussian_fraction_max, size=len(selected_peaks))

        return [multiplet_data]