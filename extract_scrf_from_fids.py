import os
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import nmrglue as ng
import pandas as pd
from pathlib import Path
from tqdm import tqdm


def run():
    # ===== Wczytywanie zmiennych środowiskowych =====
    data_dir = os.getenv("DATA_DIR", "./SCRF_extraction")
    opti_fid_path = os.getenv("OPTI_FID_PATH", "./OPTI_INPUT_SPECTRA/shimmed_lineshape.fid")

    spectra_file = os.getenv("SPECTRA_FILE", "./deconvolved_data/scrf_workshop.npy")
    spectra_file_names = os.getenv("SPECTRA_FILE_NAMES", "./deconvolved_data/scrf_workshop_names.csv")
    opi_spectrum_file = os.getenv("OPI_SPECTRUM_FILE", "./deconvolved_data/opti_spectrum.npy")
    responses_file = os.getenv("RESPONSES_FILE", "./deconvolved_data/scrf_workshop.pt")
    losses_file = os.getenv("LOSSES_FILE", "./deconvolved_data/losses_scrf_workshop.pt")

    # Tworzenie katalogów na wyniki
    for file_path in [spectra_file, spectra_file_names, opi_spectrum_file, responses_file]:
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)

    # ===== Ustawienia =====
    ph0_correction = -190.49
    ph1_correction = 0
    autophase_fn = "acme"
    target_length = None

    calibration_peak_center = "auto"
    calibration_window_halfwidth = 128
    steps = 6000
    kernel_size = 61
    kernel_sqrt = False

    # ===== Funkcje pomocnicze =====
    def fid_to_spectrum(varian_fid_path, ph0_correction, ph1_correction, autophase_fn, target_length=None, sin_pod=False):
        dic, data = ng.varian.read(varian_fid_path)
        data[0] *= 0.5
        if sin_pod:
            data = ng.proc_base.sp(data, end=0.98)

        if target_length is not None:
            pad_length = target_length - len(data)
            if pad_length > 0:
                data = ng.proc_base.zf(data, pad_length)
            else:
                data = data[:target_length]

        spec = ng.proc_base.fft(data)
        spec = ng.process.proc_autophase.autops(
            spec, autophase_fn, p0=ph0_correction, p1=ph1_correction, disp=False
        )
        return spec

    def fit_kernel(base, target, kernel_size, kernel_sqrt=True, steps=20000, verbose=False):
        kernel = torch.ones((1, 1, kernel_size), dtype=base.dtype)
        if kernel_sqrt:
            kernel /= torch.sqrt(torch.sum(kernel**2))
        else:
            kernel /= kernel_size
        kernel.requires_grad = True

        optimizer = torch.optim.Adam([kernel])

        for epoch in range(steps):
            if kernel_sqrt:
                spe_est = torch.conv1d(base, kernel**2, padding="same")
            else:
                spe_est = torch.conv1d(base, kernel, padding="same")
            loss = torch.mean(abs(target - spe_est) ** 2)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if kernel_sqrt:
            return kernel.detach() ** 2, loss.item()
        else:
            return kernel.detach(), loss.item()

    # ===== Przetwarzanie danych =====
    opti_spectrum_full = fid_to_spectrum(opti_fid_path, ph0_correction, ph1_correction, autophase_fn, target_length)
    if calibration_peak_center == "auto":
        calibration_peak_center = np.argmax(abs(opti_spectrum_full))
    fitting_range = (
        calibration_peak_center - calibration_window_halfwidth,
        calibration_peak_center + calibration_window_halfwidth + 1,
    )

    opti_spectrum = opti_spectrum_full[fitting_range[0]:fitting_range[1]]
    np.save(opi_spectrum_file, opti_spectrum)
    print(f"Optimal spectrum extracted to {opi_spectrum_file}")

    # Widma z katalogu
    spec_list = []
    spec_names = []
    print("Extracting spectra from fids...")
    for fid_path in tqdm(list(Path(data_dir).rglob("*.fid"))):
        spec = fid_to_spectrum(fid_path, ph0_correction, ph1_correction, autophase_fn, target_length=target_length)
        spec = spec[fitting_range[0]:fitting_range[1]]
        spec_list.append(spec)
        spec_names.append(fid_path.name)

    total = np.array(spec_list)
    np.save(spectra_file, total)
    pd.DataFrame(spec_names).to_csv(spectra_file_names, header=False)
    print(f"Spectra extracted to {spectra_file}")

    # SCRF
    responses = torch.empty(len(total), 1, 1, 1, 1, kernel_size)
    losses = torch.empty(len(total))
    base = torch.tensor(opti_spectrum.real).unsqueeze(0)
    targets = torch.tensor(total.real)

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

    # ===== PROSTE WYŚWIETLANIE: SCRF + widma z total.npy =====
    try:
        scrfs_loaded = torch.load(responses_file)
        if isinstance(scrfs_loaded, torch.Tensor):
            scrfs = scrfs_loaded.detach().cpu().numpy()
        else:
            scrfs = np.array(scrfs_loaded)
        scrfs = np.squeeze(scrfs)
        print(f"[INFO] SCRFs loaded: shape={scrfs.shape}")
    except Exception as e:
        raise RuntimeError(f"SCRF loading failed: {e}")

    try:
        spectra = np.load(spectra_file, allow_pickle=True)
        print(f"[INFO] Spectra loaded: shape={spectra.shape}")
    except Exception as e:
        raise RuntimeError(f"total.npy loading failed: {e}")

    n_scrf = scrfs.shape[0]
    n_spec = spectra.shape[0]
    n_show = min(10, n_scrf, n_spec)

    if n_show == 0:
        raise ValueError("No data to display.")

    indices = random.sample(range(n_spec), n_show)

    fig, axes = plt.subplots(n_show, 2, figsize=(12, 3 * n_show))

    for row, idx in enumerate(indices):
        # lewa kolumna = SCRF
        axes[row, 0].plot(scrfs[idx].ravel())
        axes[row, 0].set_title(f"SCRF #{idx}")
        axes[row, 0].set_xlabel(" ")
        axes[row, 0].set_ylabel(" ")

        # prawa kolumna = widmo
        spec = np.real(spectra[idx])
        axes[row, 1].plot(spec)
        axes[row, 1].set_title(f"Spectrum #{idx}")
        axes[row, 1].set_xlabel("Points")
        axes[row, 1].set_ylabel(" ")

    plt.tight_layout()
    plt.show()

    return

