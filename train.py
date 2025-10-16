import torch, torchaudio
import numpy as np
from pathlib import Path
from omegaconf import OmegaConf
from hydra.utils import instantiate
import datetime
import sys
import matplotlib.pyplot as plt


import matplotlib
matplotlib.use('Agg')

# silent deprecation_warning() from datapipes
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module='torchdata')

# from shiment import models
from shimnet.generators import get_datapipe

device = 'cuda' if torch.cuda.is_available() else 'cpu'
if len(sys.argv) < 2:
    print("Please provide the run directory as an argument.")
    sys.exit(1)
run_dir = Path(sys.argv[1])

config = OmegaConf.load(run_dir / "config.yaml")

if (run_dir / "train.txt").is_file():
    minimum = np.min(np.loadtxt(run_dir / "train.txt")[:,2])
else:
    minimum = float("inf")

# prepare spectra for evaluation
extra_spectra_for_evaluation = {}
frq_step = config.data.get("frq_step") or config.metadata.get("frq_step")
model_ppm_per_point = frq_step / config.metadata.spectrometer_frequency

for spectra_data in config.logging.get('extra_spectra_for_evaluation', []):
    spectrum_file = Path(spectra_data.path)
    spectrum_freqs_input_ppm, spectrum = np.loadtxt(spectrum_file).T

    spectrometer_frequency = spectra_data.get("spectrometer_frequency")
    if spectrometer_frequency is None: # spectrometer frequency unknown, assume the same as the model
        spectrum_freqs = spectrum_freqs_input_ppm
    else:
        spectrum_freqs_model_ppm = spectrum_freqs_input_ppm * spectrometer_frequency / config.metadata.spectrometer_frequency
        spectrum_freqs = np.arange(spectrum_freqs_model_ppm.min(), spectrum_freqs_model_ppm.max(), model_ppm_per_point)
        spectrum = np.interp(spectrum_freqs, spectrum_freqs_model_ppm, spectrum)

    extra_spectra_for_evaluation[Path(spectrum_file).stem] = {
        'frequencies': spectrum_freqs,
        'spectrum': spectrum,
    }

# initialization        
model = instantiate({"_target_": f"shimnet.models.{config.model.name}", **config.model.kwargs}).to(device)
model_weights_file = run_dir / f'model.pt'
optimizer = torch.optim.Adam(model.parameters())
optimizer_weights_file = run_dir / f'optimizer.pt'

def evaluate_model(stage=0, epoch=0):
    plot_dir = run_dir / "plots" / f"{stage}_{epoch}"
    plot_dir.mkdir(exist_ok=True, parents=True)

    torch.save(model.state_dict(), plot_dir / "model.pt")
    torch.save(optimizer.state_dict(), plot_dir / "optimizer.pt")
    
    num_plots = config.logging.num_plots
    pipe = get_datapipe(
            **config.data,
            include_response_function=True,
            batch_size=num_plots
        )
    batch = next(iter(pipe))

    with torch.no_grad():
        out = model(batch['noised_spectrum'].to(device))
        noised_est = torchaudio.functional.convolve(out['denoised'], out['response'].flip(dims=(-1,)).unsqueeze(1), mode="same").cpu()

        for i in range(num_plots):
            plt.figure(figsize=(30,6))
            plt.plot(batch['theoretical_spectrum'].cpu().numpy()[i,0])
            plt.plot(out['denoised'].cpu().numpy()[i,0])
            plt.savefig(plot_dir / f"{i:03d}_spectrum_clean.png")

            plt.figure(figsize=(30,6))
            plt.plot(batch['noised_spectrum'].cpu().numpy()[i,0])
            plt.plot(noised_est.cpu().numpy()[i,0])
            plt.savefig(plot_dir / f"{i:03d}_spectrum_noise.png")

            plt.figure(figsize=(10,6))
            plt.plot(batch['response_function'].cpu().numpy()[i,0,0])
            plt.plot(out['response'].cpu().numpy()[i])
            plt.savefig(plot_dir / f"{i:03d}_response.png")

            if "attention" in out:
                plt.figure(figsize=(10, 6))
                plt.plot(out['attention'].cpu().numpy()[i])
                plt.savefig(plot_dir / f"{i:03d}_attention.png")
            
            plt.close("all")

    # evaluate extra spectra
    if len(extra_spectra_for_evaluation) > 0:
        extra_spectra_dir = plot_dir / "extra_spectra"
        extra_spectra_dir.mkdir(exist_ok=True, parents=True)
    for spectrum_name, spectrum_data in extra_spectra_for_evaluation.items():
        spectrum = torch.tensor(spectrum_data['spectrum']).to(device)
        with torch.no_grad():
            out = model(spectrum.unsqueeze(0))
            noised_est = torchaudio.functional.convolve(out['denoised'], out['response'].flip(dims=(-1,)).unsqueeze(1), mode="same").cpu().squeeze(0)

        plt.figure(figsize=(30,6))
        plt.plot(spectrum_data['frequencies'], spectrum.cpu().numpy())
        plt.plot(spectrum_data['frequencies'], out['denoised'].cpu().squeeze(0).numpy())
        plt.savefig(extra_spectra_dir / f"{spectrum_name}_clean.png")
        np.savetxt(extra_spectra_dir / f"{spectrum_name}_clean.csv", np.column_stack((spectrum_data['frequencies'], out['denoised'].cpu().squeeze(0).numpy())))

        plt.figure(figsize=(30,6))
        plt.plot(spectrum_data['frequencies'], spectrum.cpu().numpy())
        plt.plot(spectrum_data['frequencies'], noised_est.numpy())
        plt.savefig(extra_spectra_dir / f"{spectrum_name}_noised.png")
        np.savetxt(extra_spectra_dir / f"{spectrum_name}_noised.csv", np.column_stack((spectrum_data['frequencies'], noised_est.numpy())))

        # Save response and attention if available
        if 'response' in out:
            plt.figure(figsize=(10,6))
            plt.plot(out['response'].cpu().squeeze(0).numpy())
            plt.savefig(extra_spectra_dir / f"{spectrum_name}_response.png")
            np.savetxt(extra_spectra_dir / f"{spectrum_name}_response.csv", out['response'].cpu().squeeze(0).numpy())

        if "attention" in out:
            plt.figure(figsize=(10, 6))
            plt.plot(out['attention'].cpu().numpy())
            plt.savefig(extra_spectra_dir / f"{spectrum_name}_attention.png")
            np.savetxt(extra_spectra_dir / f"{spectrum_name}_attention.csv", out['attention'].cpu().numpy())
        
        plt.close("all")


for i_stage, training_stage in enumerate(config.training):
    if model_weights_file.is_file():
        model.load_state_dict(torch.load(model_weights_file, weights_only=True))

    if optimizer_weights_file.is_file():
        optimizer.load_state_dict(torch.load(optimizer_weights_file, weights_only=True))
    optimizer.param_groups[0]['lr'] = training_stage.learning_rate
    
    pipe = get_datapipe(
        **config.data,
        include_response_function=True,
        batch_size=training_stage.batch_size
    )

    losses_history = []
    losses_history_limit = 64*100 // training_stage.batch_size
    
    last_evaluation = 0
    for epoch, batch in pipe.enumerate():
        
        # logging
        iters_done = epoch*training_stage.batch_size
        if (iters_done - last_evaluation) > config.logging.step:
            evaluate_model(i_stage, epoch)
            last_evaluation = iters_done
            
        if  iters_done > training_stage.max_iters:
            evaluate_model(i_stage, epoch)
            break
        
        # run model
        out = model(batch['noised_spectrum'].to(device))
        # calculate losses
        loss_response = torch.nn.functional.mse_loss(out['response'], batch['response_function'].squeeze(dim=(1,2)).to(device))
        loss_clean = torch.nn.functional.mse_loss(out['denoised'], batch['theoretical_spectrum'].to(device))
        noised_est = torchaudio.functional.convolve(out['denoised'], out['response'].flip(dims=(-1,)).unsqueeze(1), mode="same")
        loss_noised = torch.nn.functional.mse_loss(noised_est, batch['noised_spectrum'].to(device))
        loss = config.losses_weights.response*loss_response + config.losses_weights.clean*loss_clean + config.losses_weights.noised*loss_noised
        
        # logging
        losses_history.append(loss_clean.item())
        losses_history = losses_history[-losses_history_limit:]
        loss_avg = sum(losses_history)/len(losses_history)
        message = f"{epoch:7d} {loss:0.3e} {loss_avg:0.3e} {loss_clean:0.3e} {loss_response:0.3e} {loss_noised:0.3e}"
        # message = '%7i %.3e %.3e %.3e' % (epoch, loss, regress, classify)
        with open(run_dir / f'train.txt', 'a') as f:
            f.write(message + '\n')
        print(message, flush = True)
        
        # save best
        if loss_avg < minimum:
            minimum = loss_avg
            torch.save(model.state_dict(), model_weights_file)
            torch.save(optimizer.state_dict(),optimizer_weights_file)
        
        # update weights
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
        optimizer.zero_grad()
