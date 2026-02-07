# ShimNet
ShimNet is a data-driven AI solution to improve high-resolution nuclear magnetic resonance (NMR) spectra
distorted by the inhomogeneous magnetic field (less than optimal shimming). To use it, the experimental training data has to be collected (see **Data collection** below).
Example data can also be downloaded (see below). 

*ShimNet* paper (2025): [ShimNet: A neural network for post-acquisition improvement of NMR spectra distorted by magnetic-field inhomogeneity](https://doi.org/10.1021/acs.jpcb.5c02632)

Code version used in *ShimNet* paper (2025): https://github.com/center4ml/shimnet/releases/tag/JChemPhys_submission_2025

Web service: [![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/NMR-CeNT-UW/ShimNet-development)

**Reaction Monitoring (ShimNetV2-RM)** After fine-tuning, ShimNet may be used to monitor reaction. The procedure is described in section [Reaction monitoring](#reaction-monitoring)


## Installation

Python 3.10+ is required

You may install CPU-only version for inference only. If you need both training and inference, GPU version is strongly recommended.

In both CPU-only and GPU versions you may also install GUI (graphical user interface for inference)

- CPU-only version
  - with GUI (recommended): `pip install .[cpu,gui] --extra-index-url https://download.pytorch.org/whl/cpu`
  - without GUI: `pip install .[cpu] --extra-index-url https://download.pytorch.org/whl/cpu`
- GPU version (strongly recommended for training)
  - with GUI:  `pip install .[gpu,gui]`
  - without GUI: `pip install .[gpu]`

## Usage
To correct spectra with pre-trained weights:
1. download weights (model parameters):
```
python download_files.py
```
or directly from [Google Drive (ShimNetV2 600MHz)](https://drive.google.com/uc?export=download&id=1643Il3qgCupY0n8Mar6WBc2WVuoQRzie) and place it in `weights` directory

2. : run correction (e.g. `Azarone_X_supressed_600MHz.csv`):
```
python predict.py sample_data/Azarone_X_supressed_600MHz.csv -o output --config configs/shimnetV2_600.yaml --weights weights/shimnetV2_600MHz.pt
```
The output will be `output/Azarone_X_supressed_600MHz.csv_processed.csv` file

Multiple files may be processed using "*" syntax:
```
python predict.py sample_data/*600MHz.csv -o output --config configs/shimnetV2_600.yaml --weights weights/shimnetV2_600MHz.pt
```

### input format

The spectrum file for reconstruction should be in the format of two columns separated by a space and without the sign at the end of the line at the end of the file. The first column is frequency in ppm, the second is the intensity. The frequency values ​​should be in ascending order (example below):
```csv
-1.97134	0.0167137	
-1.97085	-0.00778748	
-1.97036	-0.0109595	
-1.96988	0.00825978	
-1.96939	0.0133886	
```

## Train on your data

For the model to function properly, it should be trained on calibration data from the spectrometer used for the measurements. To train a model on data from your spectrometer, please follow the instructions below.

### Training data collection

Below we describe the training data collection for Agilent/Varian spectrometers. For machines of other vendors similar procedure can be implemented.
To collect ShimNet training data use Python script (sweep_shims_lineshape_Z1Z2.py) from the calibration_loop folder to drive the spectrometer:
 1. Install TReNDS package ( trends.spektrino.com )
 2. Open VnmrJ and type: 'listenon'
 3. Put the lineshape sample (1% CHCl3 in deuterated acetone), set standard PROTON parameters, and set nt=1 (do not modify sw and at!)
 4. Shim the sample and collect the data. Save the optimally shimmed dataset
 5. Edit the sweep_shims_lineshape_Z1Z2.py script
 6. Put optimum z1 and z2 shim values as optiz1 and optiz2 below
 7. Define the calibration range as range_z1 and range_z2 (default is ok)
 8. Start the python script:
   ```
     python3 ./sweep_shims_lineshape_Z1Z2.py
   ```
   The spectrometer will start collecting spectra

### Shim Coil Response Functions (SCRF)

#### 1. Extraction

Shim Coil Response Functions (SCRF) should be extracted from the spectra with `extract_scrf_from_fids.py` script.
```
python extract_scrf_from_fids.py
```

The script uses hardcoded paths to the NMR signals (fid-s) in Agilent/Varian format: a directory with optimal measurement (`opti_fid_path` available) and a directory with calibration loop measurements (`data_dir`):
```python
# input
data_dir = "../../sample_run/loop"
opti_fid_path = "../../sample_run/opti.fid"

```

The output files are also hardcoded:
```python
# output
spectra_file = "../../sample_run/total.npy"
spectra_file_names = "../../sample_run/total.csv"
opi_spectrum_file = "../../sample_run/opti.npy"
responses_file = "../../sample_run/scrf_61.pt"
```
where only the `responses_file` is used in ShimNet training.

If the measurements are stored in a format other than Varian, you may need to change this line:
```python
dic, data = ng.varian.read(varian_fid_path)
```
(see nmrglue package documentation for details)

#### 2. Smoothing

Exctracted response functions may be noisy. In order to increase robustness, smoothing is recommended. Example code is stored in `preprocessing/smooth_SCRFs.ipynb`

### Training

1. Download multiplets database:
    ```
    python download_files.py --multiplets
    ```
2. Configure run:
  - create a run directory, e.g. `runs/my_lab_spectrometer_2025`
  - create a configuration file:
    - copy `configs/shimnetV2_600.yaml` to the run directory and rename it to `config.yaml`
       ```bash
       cp configs/configs/shimnetV2_600.yaml runs/my_lab_spectrometer_2025/config.yaml
       ```
    - replace response function paths in the config file:
       ```yaml
        response_files:
          - data/smoothed_scrf_kernels/scrf_81_600MHz_smoothed_1-1-1.pt
          - data/smoothed_scrf_kernels/scrf_81_600MHz_smoothed_1-2-1.pt
          - data/smoothed_scrf_kernels/scrf_81_600MHz_smoothed_1-4-1.pt
          - data/smoothed_scrf_kernels/scrf_81_600MHz_smoothed_1-3-3-1.pt
        ```
       e.g.
       ```yaml
         response_files:
         - ../../sample_run/scrf_61.pt
       ```
    - adjust spectrometer frequency step `frq_step` in metadata to match your data (spectrometer range in Hz divided by number of points in spectrum):
        ```yaml
        frq_step: 0.30048
        ```
    - adjust spectromer frequency in the metadata
        ```yaml
        metadata: # additional metadata, not used in the training process
          spectrometer_frequency: 700.0 # MHz
        ```
    - you may add experimental spectra as `.csv` which you want to monitor during training (to avoid overfitting):
        ```yaml
          extra_spectra_for_evaluation:
          - path: ../path/to/spectrum1.csv
          - path: ../path/to/spectrum2.csv
          ```
  - If you want to use the pre-trained model as the starting point, copy weights to the run directory and rename to `model.pt`
    ```bash
    cp weights/shimnetV2_600MHz.pt runs/my_lab_spectrometer_2025/model.pt
    ```
3. Run training:
    ```
    python train.py runs/my_lab_spectrometer_2025
    ```
    Training results will appear in `runs/my_lab_spectrometer_2025` directory.
    Model parameters are stored in `runs/my_lab_spectrometer_2025/model.pt` file
4. Use trained model:

    use `--config runs/my_lab_spectrometer_2025/config.yaml` and  `--weights runs/my_lab_spectrometer_2025/model.pt` flags, e.g.
    ```
    python predict.py my_sample1.csv -o my_output --config runs/my_lab_spectrometer_2025/config.yaml --weights runs/my_lab_spectrometer_2025/model.pt
    ```

## GUI

### Launching the GUI

The ShimNet GUI is built using Gradio. To start the application, run:

```bash
python predict-gui.py
```

Once the application starts, open your browser and navigate to:

```
http://127.0.0.1:7860
```

to access the GUI locally.

### Sharing the GUI

To make the GUI accessible over the internet, use the `--share` flag:

```bash
python predict-gui.py --share
```

A public web address will be displayed in the terminal, which you can use to access the GUI remotely or share with others.

### GUI inference with Docker

Create docker image:
```bash
docker build -t shimnetgui .
```

Run the container:
```bash
docker run -it -p 7860:7860 shimnetgui
```

The GUI should be working at `http://127.0.0.1:7860`

## Reaction monitoring

### Data preparation

1. Collect well-shimed spectra, e.g. of pre- and post- reaction mixture

2. Fit peaks with Lorentzian/Gaussian curves e.g. using MestreNova

3. Store peaks data in file(-s) as in `data/reaction_monitoring`. Notebook `preprocessing/parse_mnova_data.ipynb` may be a useful reference for creating dedicated code.

### Training

1. Prepare config:
  - Create the run dir and config:
    ```
    mkdir -p runs/mono-click_finetune
    cp configs/shimnetV2RM_mono-click_finetune.yaml runs/mono-click_finetune/config.yaml
    ```
  - Adjust paths to peak data, if needed:
    ```yaml
          singlets_files:
          - data/reaction_monitoring/mono-click_substrats-and-post-reaction-mixture_filtered_squeezed.csv
    ```

2. In order to repeat reaction monitoring training with the same settings as described in our report, shim coil response functions need to be smoothed, as described in [Smoothing](#2.-smoothing) section

3. Copy weights to use the "general" ShimNetV2 as the starting point:
    ```
    cp weights/shimnetV2_600MHz.pt runs/mono-click_finetune/model.pt
    ```

4. Run training:
    ```
    python train.py runs/mono-click_finetune
    ```

### Inference

1. Download weights (if needed)
    ```
    python download_files.py
    ```
2. Correct spectrum. Example spectra from reaction monitoring are available in `sample_data/reaction_monitoring`
    ```
    python predict.py sample_data/reaction_monitoring/mono-click/monitoring/200.csv --config configs/shimnetV2RM_mono-click_finetune.yaml --weights weights/shimnetV2RM_mono-click_finetune.pt --output_dir output/mono-click
    ```
