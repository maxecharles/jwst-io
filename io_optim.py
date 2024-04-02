"""Imports"""
from io_funcs import *
from optim_funcs import *

import os

from zodiax.experimental import deserialise
import jax
from jax import numpy as np, random as jr
import equinox as eqx

import webbpsf
from amigo.core import AMIOptics, SUB80Ramp, BaseModeller
from amigo.stats import check_symmetric, check_positive_semi_definite, total_read_noise, build_cov
from amigo.detector_layers import model_ramp
from amigo.fitting import optimise

import matplotlib.pyplot as plt
import ehtplot

jax.config.update("jax_enable_x64", True)


"""Changing stuff"""
# optimisation
n_epoch = 200
L1 = 1e0

config = {
    # "distribution": opt(5e-7, 30),
    "volcanoes": opt(5e-5, 10),
    "volc_contrast": opt(1e-3, 30),
    "position": adam_opt(1e-2, 0),
    # "aberrations":  clip(opt(2e1, 100), 1),
    "log_flux": opt(2e-4, 0),
    
    # # Read noise refinement
    # "OneOnFs": clip(opt(5e0, 100), 1.0),
    # "biases": clip(opt(3e3, 100), 4.0),
}

# outputting
output_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/io/output/"

# model
ngroups = 3
model_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/bfe/"
filter_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/data/niriss_filters/"

"""Load data."""
io = DefinitelyRealIo(night_day_ratio=1, volc_contrast=1e-2, seed=2, n_volcanoes=3)


"""Set up the model."""
BFE = deserialise(model_dir + "PolyBFE_trained.zdx")
pupil_mask = deserialise(model_dir + "PolyBFE_AMI_mask.zdx")

# Get webbpsf optical system for OPD
inst = webbpsf.NIRISS()
# inst.load_wss_opd_by_date(files[0][0].header["DATE-BEG"], verbose=False)
opd = np.array(inst.get_optical_system().planes[0].opd)

# Get dLux optical system
optics = AMIOptics(
    opd=opd,
    pupil_mask=pupil_mask,
    normalise=True,
    )

detector = SUB80Ramp()

key = jr.PRNGKey(0)

params = {}
params["position"] = jr.uniform(key, (2,), minval=-0.5, maxval=0.5)
params["log_flux"] = jr.uniform(key, (1,), minval=6.9, maxval=7.1)
params["aberrations"] = 5*jr.normal(key, optics.coefficients.shape)
params["biases"] = 25*jr.normal(key, (80, 80)) + 80
params["OneOnFs"] = 10*jr.normal(key, (ngroups, 80, 2))
params["BFE"] = BFE
# params["aberrations"] = np.zeros(optics.coefficients.shape)
# params["biases"] = np.zeros((80, 80))
# params["OneOnFs"] = np.zeros((ngroups, 80, 2))
# params["BFE"] = PolyBFE(ksize=5, oversample=1, orders=[1, 2])
params["filter"] = get_filter_spectrum("F430M", file_path=filter_dir)
params["optics"] = optics
params["detector"] = detector
params["source"] = IoSource(
    position=params["position"],
    log_flux=params["log_flux"],
    distribution=io.data,
    spectrum=params["filter"],
)
params["volcanoes"] = io.volcanoes / io.volcanoes.sum() # DO THIS IN THE CLASS
params["disk"] = io.disk
params["volc_contrast"] = 5e-2
true_model = BaseModeller(params)

def model_fn(model, n_groups=ngroups):

    source = model.source.set(
        ["position", "log_flux"],
        [model.position, model.log_flux],
    )

    distribution = model.disk.data + model.volc_contrast * model.volcanoes
    source = source.set('distribution', distribution)  # ONLY FIT THE VOLCANOES

    # # Apply correct aberrations
    # optics = model.optics.set("coefficients", model.aberrations)

    # Make sure this has correct position units and get wavefronts
    PSF = source.model(optics, return_psf=True)

    # Apply the detector model and turn it into a ramp
    psf = model.detector.model(PSF)
    ramp = model_ramp(psf, n_groups)

    # Now apply the CNN BFE and downsample
    ramp = eqx.filter_vmap(model.BFE.apply_array)(ramp)
    ramp = vmap(dlu.resize, (0, None))(ramp, 80)

    # Apply bias and one of F correction
    ramp += total_read_noise(model.biases, model.OneOnFs)
    return ramp


"""Simulating data"""
clean_int = model_fn(true_model, n_groups=ngroups)
data = add_noise_to_ramp(clean_int)  # adding photon noise to ramp
io_mask = np.where(initialise_disk()==0, False, True)

"""Optimising"""
initial_model = true_model.set(
    [
        "position",
        "log_flux",
        "distribution",
        "volcanoes",
        "volc_contrast",
        "aberrations",
        "biases",
        "OneOnFs",
    ],
    [
        true_model.position + 0.1*jr.normal(key, shape=(2,)),
        7.,
        initialise_disk(normalise=True),
        initialise_disk(normalise=True),
        0.95*true_model.volc_contrast,
        np.zeros_like(true_model.aberrations),
        np.zeros_like(true_model.biases),
        np.zeros_like(true_model.OneOnFs)
    ],
)

# Optimisation
final_model, losses, params_out, opt_state = optimise(
    initial_model,
    {"model_fn": model_fn, "data": data, "mask": io_mask, "L1": L1},
    loss_fn,
    n_epoch,
    config,
    norm_fn=norm_fn,
    # grad_fn=grad_fn,
    print_grads=False,
    return_state=True,
    nan_method="none",
)

"""Plotting"""
now = lambda: datetime.datetime.now().strftime("%H.%M.%S_%d:%m:%Y")
save_path = output_dir + now() + "/"

if not os.path.exists(save_path):
    os.makedirs(save_path)

plot_params(true_model, losses, params_out, format_fn, save=save_path)

plot_io_with_truth(
    final_model,
    model_fn,
    data,
    losses,
    ngroups,
    opt_state,
    initial_distribution=initial_model.source.distribution,
    truth=true_model.distribution,
    save=save_path,
    )
