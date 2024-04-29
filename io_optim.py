"""Imports"""
from io_funcs import DefinitelyRealIo, get_filter_spectrum, IoSource, add_noise_to_ramp, initialise_disk
from io_funcs import io_model_fn as model_fn
from optim_funcs import opt, adam_opt, loss_fn, norm_fn
from plotting import plot_params, plot_io_with_truth

import os
import datetime

from zodiax.experimental import deserialise
import jax
from jax import numpy as np, random as jr, vmap
import equinox as eqx
import dLux.utils as dlu

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
n_epoch = 500
L1 = 0e0

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
ngroups = 500
model_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/bfe/"
filter_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/data/niriss_filters/"

"""Saving stuff"""
def save_script_to_txt(save_dir: str):
    # Get the path of the current Python script
    script_path = os.path.abspath(__file__)

    # Open the script file for reading
    with open(script_path, 'r') as script_file:
        script_content = script_file.read()

    # Define the path for the output text file (e.g., 'script.txt')
    output_file_path = save_dir + 'script.txt'

    # Write the script content to the output text file
    with open(output_file_path, 'w') as output_file:
        output_file.write(script_content)
        
now = lambda: datetime.datetime.now().strftime("%H.%M.%S_%d:%m:%Y")
save_dir = output_dir + now() + "/"

# creating save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_script_to_txt(save_dir)  # saving copy of script to save directory

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
plot_params(np.array(losses), params_out, save=save_dir, true_model=true_model)

plot_io_with_truth(
    final_model,
    model_fn,
    data,
    losses,
    ngroups,
    opt_state,
    initial_distribution=initial_model.source.distribution,
    truth=true_model.distribution,
    save=save_dir,
    )
