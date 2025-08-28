import nmrglue as ng

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