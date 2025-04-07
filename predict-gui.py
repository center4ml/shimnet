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

def fast_normalize(x):
    return x / np.max(x)

def process_file(input_file, config_file, weights_file, input_spectrometer_frequency=None,reference_spectra=[], normalize_spectra_for_plotting=True):
    if input_spectrometer_frequency == 0:
        input_spectrometer_frequency = None
    # Load configuration and initialize predictor
    config = OmegaConf.load(config_file.name)
    model_ppm_per_point = config.data.frq_step / config.metadata.spectrometer_frequency
    predictor = initialize_predictor(config, weights_file.name)

    # Load input data
    input_data = np.loadtxt(input_file.name)
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
    output_file = f"{Path(input_file.name).stem}_processed{Path(input_file.name).suffix}"
    np.savetxt(output_file, output_data)

    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=input_freqs_input_ppm, y=fast_normalize(input_spectrum) if normalize_spectra_for_plotting else input_spectrum, mode='lines', name='Input Spectrum'))
    fig.add_trace(go.Scatter(x=input_freqs_input_ppm, y=fast_normalize(output_prediction) if normalize_spectra_for_plotting else output_prediction, mode='lines', name='Corrected Spectrum'))
    for reference_spectrum_file in reference_spectra:
        reference_data = np.loadtxt(reference_spectrum_file.name)
        fig.add_trace(go.Scatter(x=reference_data[:, 0], y=fast_normalize(reference_data[:, 1]) if normalize_spectra_for_plotting else reference_data[:, 1], mode='lines', name=f'Reference Spectrum {Path(reference_spectrum_file.name).stem}'))
    fig.update_layout(title="Spectrum Visualization", xaxis_title="Frequency (ppm)", yaxis_title="Intensity")

    return fig, output_file

# app = gr.Interface(
#     fn=process_file,
#     inputs=[
#         gr.File(label="Input File (.txt | .csv)"),
#         gr.File(label="Config File (.yaml)"),
#         gr.File(label="Weights File (.pt)"),
#         gr.Number(label="Input Spectrometer Frequency (MHz)", value=None)
#     ],
#     outputs=[
#         gr.Plot(label="Spectrum Visualization"),
#         gr.File(label="Download Processed File")
#     ],
#     title="NMR Spectrum Prediction",
#     description="Upload your input file, configuration, and weights to process the NMR spectrum."
# )

# Gradio app
with gr.Blocks() as app:
    gr.Markdown("# ShimNet Spectra Correction")
    gr.Markdown("Upload your input file, configuration, and weights to process the NMR spectrum.")

    with gr.Row():
        with gr.Column():
            config_file = gr.File(label="Config File (.yaml)", height=120, value="configs/shimnet_600.yaml")
            weights_file = gr.File(label="Weights File (.pt)", height=120, value="weights/shimnet_600MHz.pt")
        
        with gr.Column():
            input_file = gr.File(label="Input File (.txt | .csv)", height=120)
            input_spectrometer_frequency = gr.Number(label="Input Spectrometer Frequency (MHz) (0 or empty if the same as in the loaded model)", value=None)
            gr.Markdown("Upload reference spectra files (optional). Reference spectra will be plotted for comparison.")
            reference_spectra = gr.Files(label="Reference Spectra File(s) (.txt | .csv)", height=120)
    normalize_spectra_for_plotting = gr.Checkbox(label="Normalize Spectra for Plotting", value=True)
    process_button = gr.Button("Process File")
    plot_output = gr.Plot(label="Spectrum Visualization")
    download_button = gr.File(label="Download Processed File", interactive=False, height=120)

    process_button.click(
        process_file,
        inputs=[input_file, config_file, weights_file, input_spectrometer_frequency, reference_spectra, normalize_spectra_for_plotting],
        outputs=[plot_output, download_button]
    )

app.launch(share=True)