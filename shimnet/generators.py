import numpy as np
import torch
import torchdata
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod

# from itertools import islice

def random_value(min_value, max_value, generator=None):
    return (min_value + torch.rand(1, generator=generator) * (max_value - min_value)).item()

def random_loguniform(min_value, max_value, generator=None):
    return (min_value * torch.exp(torch.rand(1, generator=generator) * (torch.log(torch.tensor(max_value)) - torch.log(torch.tensor(min_value))))).item()

def calculate_theoretical_spectrum(peaks_parameters: dict, frq_frq:torch.Tensor):
    # extract parameters
    tff_lin = peaks_parameters["tff_lin"]
    twf_lin = peaks_parameters["twf_lin"]
    thf_lin = peaks_parameters["thf_lin"]
    trf_lin = peaks_parameters["trf_lin"]

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
    tsf_frq = tsf_linfrq.sum(0, keepdim = True)
    return tsf_frq


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

class TheoreticalMultipletSpectraGenerator:
    def __init__(self, atom_groups_data_file=None, pixels=2048, frq_step=11160.7142857 / 32768,
                 number_of_signals_min=1, number_of_signals_max=8,
                 spectrum_width_min=0.2, spectrum_width_max=1, relative_width_min=1, relative_width_max=2,
                 relative_height_min=1, relative_height_max=1, relative_frequency_min=-0.4, relative_frequency_max=0.4,
                 thf_min=1/16, thf_max=16, trf_min=0, trf_max=1, multiplicity_j1_min=0, multiplicity_j1_max=15,
                 multiplicity_j2_min=0, multiplicity_j2_max=15, seed=42, **kwargs):
        # Read atom_groups_data from file
        if atom_groups_data_file is None:
            self.atom_groups_data = np.ones((1,3), dtype=int)
        else:
            self.atom_groups_data = np.atleast_2d(np.loadtxt(atom_groups_data_file, usecols=(1,2,3), dtype=int))
        self.pixels = pixels
        self.frq_step = frq_step
        self.number_of_signals_min = number_of_signals_min
        self.number_of_signals_max = number_of_signals_max
        self.spectrum_width_min = spectrum_width_min
        self.spectrum_width_max = spectrum_width_max
        self.relative_width_min = relative_width_min
        self.relative_width_max = relative_width_max
        self.relative_height_min = relative_height_min
        self.relative_height_max = relative_height_max
        self.relative_frequency_min = relative_frequency_min
        self.relative_frequency_max = relative_frequency_max
        self.thf_min = thf_min
        self.thf_max = thf_max
        self.trf_min = trf_min
        self.trf_max = trf_max
        self.multiplicity_j1_min = multiplicity_j1_min
        self.multiplicity_j1_max = multiplicity_j1_max
        self.multiplicity_j2_min = multiplicity_j2_min
        self.multiplicity_j2_max = multiplicity_j2_max
        self.frq_frq = torch.arange(-pixels // 2, pixels // 2) * frq_step
        self.rng_getter = RngGetter(seed=seed) # self.rng_getter.get_rng(seed=seed) to get random generator 

    def __call__(self, seed=None):
        rng = self.rng_getter.get_rng(seed=seed)

        spectrum, spectrum_data = generate_theoretical_spectrum(
            number_of_signals_min=self.number_of_signals_min,
            number_of_signals_max=self.number_of_signals_max,
            spectrum_width_min=self.spectrum_width_min,
            spectrum_width_max=self.spectrum_width_max,
            relative_width_min=self.relative_width_min,
            relative_width_max=self.relative_width_max,
            tff_min=self.relative_frequency_min * self.pixels * self.frq_step,
            tff_max=self.relative_frequency_max * self.pixels * self.frq_step,
            thf_min=self.thf_min,
            thf_max=self.thf_max,
            trf_min=self.trf_min,
            trf_max=self.trf_max,
            relative_height_min=self.relative_height_min,
            relative_height_max=self.relative_height_max,
            multiplicity_j1_min=self.multiplicity_j1_min,
            multiplicity_j1_max=self.multiplicity_j1_max,
            multiplicity_j2_min=self.multiplicity_j2_min,
            multiplicity_j2_max=self.multiplicity_j2_max,
            atom_groups_data=self.atom_groups_data,
            frq_frq=self.frq_frq,
            generator=rng
        )
        return spectrum, {"spectrum_data": spectrum_data, "frq_frq": self.frq_frq}

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
                 include_spectrum_data=False, include_peak_mask=False, include_response_function=False, seed=None):
        super().__init__(batch_size=batch_size, seed=seed)
        self.clean_spectra_generator = clean_spectra_generator
        self.response_generator = response_generator
        self.noise_generator = noise_generator
        self.include_spectrum_data = include_spectrum_data
        self.include_peak_mask = include_peak_mask
        self.include_response_function = include_response_function

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