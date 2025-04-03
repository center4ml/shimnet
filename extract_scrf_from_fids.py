import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import nmrglue as ng
from tqdm import tqdm

# input
data_dir = "../../../dane/600Hz_20241227/loop"
opti_fid_path = "../../../dane/600Hz_20241227/opti"

# output
spectra_file = "data/600Hz_20241227-Lnative/total.npy"
spectra_file_names = "data/600Hz_20241227-Lnative/total.csv"
opi_spectrum_file = "data/600Hz_20241227-Lnative/opti.npy"
responses_file = "data/600Hz_20241227-Lnative/scrf_61.pt"
losses_file = "data/600Hz_20241227-Lnative/losses_scrf_61.pt"

# create directories for output files
for file_path in [spectra_file, spectra_file_names, opi_spectrum_file, responses_file]:
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)

# settings: fid to spectrum
ph0_correction =-190.49
ph1_correction = 0
autophase_fn="acme"
target_length = None

# settings: SCRF extraction
calibration_peak_center = "auto"#12277 # the center of the calibration peak (peak of)
calibration_window_halfwidth = 128 # the half width of the calibration window
steps = 6000
kernel_size = 61
kernel_sqrt = False # True to allow only positive values

# fid to spectra processing

def fid_to_spectrum(varian_fid_path, ph0_correction, ph1_correction, autophase_fn, target_length=None, sin_pod=False):
    dic, data = ng.varian.read(varian_fid_path)
    data[0] *= 0.5
    if sin_pod:
        data = ng.proc_base.sp(data, end=0.98)

    if target_length is not None:
        if (pad_length := target_length - len(data)) > 0:
            data = ng.proc_base.zf(data, pad_length)
        else:
            data = data[:target_length]

    spec=ng.proc_base.fft(data)
    spec = ng.process.proc_autophase.autops(spec, autophase_fn, p0=ph0_correction, p1=ph1_correction, disp=False)

    return spec

# process optimal measurement fid to spectrum
opti_spectrum_full = fid_to_spectrum(opti_fid_path, ph0_correction, ph1_correction, autophase_fn, target_length=target_length)

if calibration_peak_center == "auto":
    calibration_peak_center = np.argmax(abs(opti_spectrum_full))
fitting_range = (calibration_peak_center - calibration_window_halfwidth, calibration_peak_center+calibration_window_halfwidth+1)

opti_spectrum = opti_spectrum_full[fitting_range[0]:fitting_range[1]]
np.save(opi_spectrum_file, opti_spectrum)
print(f"Optimal spectrum extracted to {opi_spectrum_file}")

# process loop fids to spectra
spec_list=[]
spec_names=[]

print("Extracting spectra from fids...")
for fid_path in tqdm(list(Path(data_dir).rglob('*.fid'))):
    spec = fid_to_spectrum(fid_path, ph0_correction, ph1_correction, autophase_fn, target_length=target_length)[fitting_range[0]:fitting_range[1]]
       
    spec_list.append(spec)
    spec_names.append(fid_path.name)

total = np.array(spec_list)
np.save(spectra_file, total)
pd.DataFrame(spec_names).to_csv(spectra_file_names, header=False)
# total = np.load(spectra_file)
print(f"Spectra extracted to {spectra_file}")

# process SCRF extraction
def fit_kernel(base, target, kernel_size, kernel_sqrt=True, steps=20000, verbose=False):

    kernel = torch.ones((1,1,kernel_size), dtype=base.dtype)
    if kernel_sqrt:
        kernel /= torch.sqrt(torch.sum(kernel**2))
    else:
        kernel /= kernel_size
    kernel.requires_grad = True

    optimizer = torch.optim.Adam([kernel])

    for epoch in range(steps):
        if kernel_sqrt:
            spe_est = torch.conv1d(base, kernel**2, padding='same')
        else:
            spe_est = torch.conv1d(base, kernel, padding='same')
        loss = torch.mean(abs(target - spe_est)**2) #torch.nn.functional.mse_loss(spe_est, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if verbose and (epoch+1) % 100 == 0:
            print(epoch, loss.item())
    if kernel_sqrt:
        return kernel.detach()**2, loss.item()
    else:
        return kernel.detach(), loss.item()

responses = torch.empty(len(total), 1, 1, 1, 1, kernel_size)
losses = torch.empty(len(total))
base = torch.tensor(opti_spectrum.real).unsqueeze(0)
targets = torch.tensor(total.real)

# normalization
base /= base.sum()
targets /= targets.sum(dim=(-1,), keepdim=True)

print("\nExtracting SCRFs...")
for i, target in tqdm(enumerate(targets), total=len(targets)):    
    kernel, loss = fit_kernel(base, target.unsqueeze(0), kernel_size, kernel_sqrt=kernel_sqrt, steps=steps)
    responses[i, 0, 0] = kernel
    losses[i] = loss

torch.save(responses, responses_file)
torch.save(losses, losses_file)
print(f"SCRFs extracted to {responses_file}, losses saved to {losses_file}")
