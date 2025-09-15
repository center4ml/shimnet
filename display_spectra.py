import os
import shutil
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import nmrglue as ng
from tqdm import tqdm
import plotly.graph_objects as go

def run(input_file, corrected_file, reference_file):

    # ======= WCZYTANIE DANYCH =======
    input_spectrum_file = os.getenv("INPUT_SPECTRUM_FILE", input_file)
    corrected_spectrum_file = os.getenv("CORRECTED_SPECTRUM_FILE", corrected_file)
    reference_spectrum_file = os.getenv("REFERENCE_SPECTRUM_FILE", reference_file)

    def load_spectrum(path):
        df = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
        df = df.dropna(axis=1, how="all")  # usuń puste kolumny
        freqs = df.iloc[:, 0].values.astype(float)
        intensities = df.iloc[:, 1].values.astype(float)
        return freqs, intensities

    freqs_input, spec_input = load_spectrum(input_spectrum_file)
    freqs_corr, spec_corr = load_spectrum(corrected_spectrum_file)
    freqs_ref, spec_ref = load_spectrum(reference_spectrum_file)

    # ======= NORMALIZACJA =======
    norm_val = spec_input.max()
    spec_input /= norm_val
    spec_corr /= norm_val
    spec_ref /= norm_val

    # ======= WYKRES =======
    fig = go.Figure()

    # Input spectrum
    fig.add_trace(go.Scatter(
        x=freqs_input,
        y=spec_input,
        mode='lines',
        name='Input Spectrum',
        line=dict(color='#EF553B')
    ))

    # Corrected spectrum
    fig.add_trace(go.Scatter(
        x=freqs_corr,
        y=spec_corr,
        mode='lines',
        name='Corrected Spectrum',
        line=dict(color='#00cc96')
    ))

    # Reference spectrum
    fig.add_trace(go.Scatter(
        x=freqs_ref,
        y=spec_ref,
        mode='lines',
        name='Reference Spectrum',
        line=dict(color='#636efa')
    ))

    # ======= USTAWIENIA WYKRESU =======
    fig.update_layout(
        title="Spectrum Visualization",
        xaxis_title="Frequency (ppm)",
        yaxis_title=" ",
        legend=dict(
            title="Spectra",
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # odwrócenie osi X
    fig.update_xaxes(autorange="reversed")

    # ======= POKAŻ WYKRES =======
    fig.show()
    return


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=False)
    parser.add_argument("--corrected", type=str, required=False)
    parser.add_argument("--reference", type=str, required=False)
    args = parser.parse_args()
    run(args.input, args.corrected, args.reference)

