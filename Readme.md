# ShimNet
ShimNet is a data-driven AI solution to improve high-resolution nuclear magnetic resonance (NMR) spectra
distorted by the inhomogeneous magnetic field (less than optimal shimming). To use it, the experimental training data has to be collected (see **Data collection** below).
Example data can also be downloaded (see below). 

Paper: [ShimNet: A neural network for post-acquisition improvement of NMR spectra distorted by magnetic-field inhomogeneity](https://chemrxiv.org/engage/chemrxiv/article-details/67ef866)

Web service: [![Open in Hugging Face Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/NMR-CeNT-UW/ShimNet)

## Installation

Python 3.9+ (3.10+ for GUI)

GPU version (for training and inference)
```
pip install -r requirements-gpu.txt
```

CPU version (for inference, not recommended for training)
```
pip install -r requirements-cpu.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Usage
To correct spectra presented in the paper:
1. download weights (model parameters):
```
python download_files.py
```
or directly from [Google Drive 700MHz](https://drive.google.com/uc?export=download&id=17fTNWl7YW6mPbbZWga0EfdoF_6S8fCke) and [Google Drive 600MHz](https://drive.google.com/uc?export=download&id=1_VxOpFGJcFsOa5DHOW2GJbP8RvHCmC1N) and place it in `weights` directory


2. : run correction (e.g. `Azarone_20ul_700MHz.csv`):
```
python predict.py sample_data/Azarone_20ul_700MHz.csv -o output --config configs/shimnet_700.yaml --weights weights/shimnet_700MHz.pt
```
The output will be `output/Azarone_20ul_700MHz_processed.csv` file

Multiple files may be processed using "*" syntax:
```
python predict.py sample_data/*700MHz.csv -o output --config configs/shimnet_700.yaml --weights weights/shimnet_700MH
z.pt
```

For 600 MHz data use `--config configs/shimnet_600.yaml` and  `--weights weights/shimnet_600MHz.pt`, e.g.:

```
python predict.py sample_data/CresolRed_after_styrene_600MHz.csv -o output --config configs/shimnet_600.yaml --weights weights/shimnet_600MHz.pt
```

### input format

The spectrum file for reconstruction should be in the format of two columns separated by a space and without the sign at the end of the line at the end of the file(example below):
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

### SCRF extraction
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

### Training

1. Download multiplets database:
    ```
    python download_files.py --multiplets
    ```
2. Configure run:
  - create a run directory, e.g. `runs/my_lab_spectrometer_2025`
  - create a configuration file:
    1. copy `configs/shimnet_template.py` to the run directory and rename it to `config.yaml`
       ```bash
       cp configs/shimnet_template.py runs/my_lab_spectrometer_2025/config.yaml
       ```
    2. edit the SCRF in path in the config file:
       ```yaml
         response_functions_files:
         - path/to/srcf_file
       ```
       e.g.
       ```yaml
         response_functions_files:
         - ../../sample_run/scrf_61.pt
       ```
    3. adjust spectrometer frequency step `frq_step` to match your data (spectrometer range in Hz divided by number of points in spectrum):
        ```yaml
        frq_step: 0.34059797
        ```
    4. adjust spectromer frequency in the metadata
        ```yaml
        metadata: # additional metadata, not used in the training process
          spectrometer_frequency: 700.0 # MHz
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

## Repeat training on our data

If you want to train the network using the calibration data from our paper, follow the procedure below.

1. Download multiplets database and our SCRF files:
    ```
    python download_files.py --multiplets --SCRF --no-weights
    ```
    or directly download from Google Drive and store in `data/` directory: [Response Functions 600MHz](https://drive.google.com/file/d/1J-DsPtaITXU3TFrbxaZPH800U1uIiwje/view?usp=sharing), [Response Functions 700MHz](https://drive.google.com/file/d/113al7A__yYALx_2hkESuzFIDU3feVtNY/view?usp=sharing), [Multiplets data](https://drive.google.com/file/d/1QGvV-Au50ZxaP1vFsmR_auI299Dw-Wrt/view?usp=sharing)

2. Configure run
    - For 600MHz spectrometer:
      ```bash
      mkdir -p runs/repeat_paper_training_600MHz
      cp configs/shimnet_600.yaml runs/repeat_paper_training_600MHz/config.yaml
      ```
    - For 700 MHz spectrometer:
      ```bash
      mkdir -p runs/repeat_paper_training_700MHz
      cp configs/shimnet_700.yaml runs/repeat_paper_training_700MHz/config.yaml
      ```
3. Run training:
    ```
    python train.py runs/repeat_paper_training_600MHz
    ```
    or
    ```
    python train.py runs/repeat_paper_training_700MHz
    ```
    Training results will appear in `runs/repeat_paper_training_600MHz` or `runs/repeat_paper_training_700MHz` directory.

## GUI

### Installation

To use the ShimNet GUI, ensure you have Python 3.10 installed (not tested with Python 3.11+). After installing the ShimNet requirements (CPU/GPU), install the additional dependencies for the GUI:

```bash
pip install -r requirements-gui.txt
```

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
