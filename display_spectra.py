import pandas as pd
import plotly.graph_objects as go

def load_spectrum(path):
    df = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
    df = df.dropna(axis=1, how="all")  # usuń puste kolumny
    freqs = df.iloc[:, 0].values.astype(float)
    intensities = df.iloc[:, 1].values.astype(float)
    return freqs, intensities


def display_spectra(input_file, corrected_file1, reference_file=None, corrected_file2=None):
    freqs_input, spec_input = load_spectrum(input_file)
    freqs_corr1, spec_corr1 = load_spectrum(corrected_file1)

    plot_currect_file2 = corrected_file2 is not None
    plot_reference_file = reference_file is not None
    if plot_currect_file2:
        freqs_corr2, spec_corr2 = load_spectrum(corrected_file2)
    if plot_reference_file:
        freqs_ref, spec_ref = load_spectrum(reference_file)

    # ======= NORMALIZACJA =======
    norm_val = spec_input.max()
    spec_input /= norm_val
    spec_corr1 /= norm_val
    if plot_currect_file2:
        spec_corr2 /= norm_val
    if plot_reference_file:
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

    # Corrected spectrum1
    fig.add_trace(go.Scatter(
        x=freqs_corr1,
        y=spec_corr1,
        mode='lines',
        name='Corrected Spectrum 1',
        line=dict(color='#00cc96')
    ))

    # Corrected spectrum2
    if plot_currect_file2:
        fig.add_trace(go.Scatter(
            x=freqs_corr2,
            y=spec_corr2,
            mode='lines',
            name='Corrected Spectrum 2',
            line=dict(color='#ff6700')
        ))

    # Reference spectrum
    if plot_reference_file:
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
    display_spectra(args.input, args.corrected, args.reference)

