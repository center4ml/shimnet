import torch
torch.set_grad_enabled(False)
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
import gradio as gr
import plotly.graph_objects as go

from src.models import ShimNetWithSCRF, Predictor
from predict import Defaults, resample_input_spectrum, resample_output_spectrum, initialize_predictor

# silent deprecation warnings
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

import argparse

# Add argument parsing for server_name
parser = argparse.ArgumentParser(description="Launch ShimNet Spectra Correction App")
parser.add_argument(
    "--server_name",
    type=str,
    default="127.0.0.1",
    help="Server name to bind the app (default: 127.0.0.1). Use 0.0.0.0 for external access."
)
parser.add_argument(
    "--share",
    action="store_true",
    help="If set, generates a public link to share the app."
)
args = parser.parse_args()

def process_file(input_file, config_file, weights_file, input_spectrometer_frequency=None,reference_spectrum=None):
    if input_spectrometer_frequency == 0:
        input_spectrometer_frequency = None
    # Load configuration and initialize predictor
    config = OmegaConf.load(config_file)
    model_ppm_per_point = config.data.frq_step / config.metadata.spectrometer_frequency
    predictor = initialize_predictor(config, weights_file)

    # Load input data
    input_data = np.loadtxt(input_file)
    input_freqs_input_ppm, input_spectrum = input_data[:, 0], input_data[:, 1]

    # Convert input frequencies to model's frequency
    if input_spectrometer_frequency is not None:
        input_freqs_model_ppm = input_freqs_input_ppm * input_spectrometer_frequency / config.metadata.spectrometer_frequency
    else:
        input_freqs_model_ppm = input_freqs_input_ppm

    # Resample input spectrum
    freqs, spectrum = resample_input_spectrum(input_freqs_model_ppm, input_spectrum, model_ppm_per_point)

    # Scale and process spectrum
    spectrum_tensor = torch.tensor(spectrum).float()
    scaling_factor = Defaults.SCALE / spectrum_tensor.max()
    spectrum_tensor *= scaling_factor
    prediction = predictor(spectrum_tensor).numpy()
    prediction /= scaling_factor

    # Resample output spectrum
    output_prediction = resample_output_spectrum(input_freqs_model_ppm, freqs, prediction)

    # Prepare output data for download
    output_data = np.column_stack((input_freqs_input_ppm, output_prediction))
    output_file = f"{Path(input_file).stem}_processed{Path(input_file).suffix}"
    np.savetxt(output_file, output_data)

    # Create Plotly figure
    fig = go.Figure()

    # Add Input Spectrum and Corrected Spectrum (always visible)
    normalization_value = input_spectrum.max()
    fig.add_trace(go.Scatter(x=input_freqs_input_ppm, y=input_spectrum/normalization_value, mode='lines', name='Input Spectrum', visible=True, line=dict(color='#EF553B'))) # red
    fig.add_trace(go.Scatter(x=input_freqs_input_ppm, y=output_prediction/normalization_value, mode='lines', name='Corrected Spectrum', visible=True, line=dict(color='#00cc96'))) # green

    if reference_spectrum is not None:
        reference_spectrum_freqs, reference_spectrum_intensity = np.loadtxt(reference_spectrum).T
        reference_spectrum_intensity /= reference_spectrum_intensity.max()
        n_zooms = 50
        zooms = np.geomspace(0.01, 100, 2 * n_zooms + 1)

        # Add Reference Data traces (initially invisible)
        for zoom in zooms:
            fig.add_trace(
                go.Scatter(
                    x=reference_spectrum_freqs,
                    y=reference_spectrum_intensity * zoom,
                    mode='lines',
                    name=f'Reference Data (Zoom: {zoom:.2f})',
                    visible=False,
                    line=dict(color='#636efa')
                )
            )
        # Make the middle zoom level visible by default
        fig.data[2 * n_zooms // 2 + 2].visible = True

        # Create and add slider
        steps = []
        for i in range(2, len(fig.data)):  # Start from the reference data traces
            step = dict(
                method="update",
                args=[{"visible": [True, True] + [False] * (len(fig.data) - 2)}],  # Keep first two traces visible
            )
            step["args"][0]["visible"][i] = True  # Toggle i'th reference trace to "visible"
            steps.append(step)

        sliders = [dict(
            active=n_zooms,
            currentvalue={"prefix": "Reference zoom: "},
            pad={"t": 50},
            steps=steps
        )]

        fig.update_layout(
            sliders=sliders
        )

    fig.update_layout(
        title="Spectrum Visualization",
        xaxis_title="Frequency (ppm)",
        yaxis_title="Intensity"
    )

    return fig, output_file

# Gradio app
with gr.Blocks() as app:
    gr.Markdown("# ShimNet Spectra Correction")
    gr.Markdown("[ShimNet: A neural network for post-acquisition improvement of NMR spectra distorted by magnetic-field inhomogeneity](https://chemrxiv.org/engage/chemrxiv/article-details/67ef86686dde43c90860d315)")
    gr.Markdown("Upload your input file, configuration, and weights to process the NMR spectrum.")

    with gr.Row():
        with gr.Column():
            model_selection = gr.Radio(
                label="Select Model",
                choices=["600 MHz", "700 MHz", "Custom"],
                value="600 MHz"
            )
            config_file = gr.File(label="Custom Config File (.yaml)", visible=False, height=120)
            weights_file = gr.File(label="Custom Weights File (.pt)", visible=False, height=120)
        
        with gr.Column():
            input_file = gr.File(label="Input File (.txt | .csv)", height=120)
            input_spectrometer_frequency = gr.Number(label="Input Spectrometer Frequency (MHz) (0 or empty if the same as in the loaded model)", value=None)
            gr.Markdown("Upload reference spectrum files (optional). Reference spectrum will be plotted for comparison.")
            reference_spectrum_file = gr.File(label="Reference Spectra File (.txt | .csv)", height=120)
    
    process_button = gr.Button("Process File")
    plot_output = gr.Plot(label="Spectrum Visualization")
    download_button = gr.File(label="Download Processed File", interactive=False, height=120)

    # Update visibility of config and weights fields based on model selection
    def update_visibility(selected_model):
        if selected_model == "Custom":
            return gr.update(visible=True), gr.update(visible=True)
        else:
            return gr.update(visible=False), gr.update(visible=False)

    model_selection.change(
        update_visibility,
        inputs=[model_selection],
        outputs=[config_file, weights_file]
    )

    # Process button click logic
    def process_file_with_model(input_file, model_selection, config_file, weights_file, input_spectrometer_frequency, reference_spectrum_file):
        if model_selection == "600 MHz":
            config_file = "configs/shimnet_600.yaml"
            weights_file = "weights/shimnet_600MHz.pt"
        elif model_selection == "700 MHz":
            config_file = "configs/shimnet_700.yaml"
            weights_file = "weights/shimnet_700MHz.pt"
        else:
            config_file = config_file.name
            weights_file = weights_file.name

        return process_file(input_file.name, config_file, weights_file, input_spectrometer_frequency, reference_spectrum_file.name if reference_spectrum_file else None)

    process_button.click(
        process_file_with_model,
        inputs=[input_file, model_selection, config_file, weights_file, input_spectrometer_frequency, reference_spectrum_file],
        outputs=[plot_output, download_button]
    )

app.launch(share=args.share, server_name=args.server_name)

# '#636efa',
#  '#EF553B',
#  '#00cc96',
#  '#ab63fa',
#  '#FFA15A',
#  '#19d3f3',
#  '#FF6692',
#  '#B6E880',
#  '#FF97FF',
#  '#FECB52'
