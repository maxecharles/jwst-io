"""Imports"""
from io_funcs import (
    DefinitelyRealIo,
    # SimpleIoSource,
    ComplexIoSource,
    get_filter_spectrum,
    add_noise_to_ramp,
    initialise_disk,
    sim_io_model_fn as model_fn,
)
from optim_funcs import opt, delay, loss_fn, complex_norm_fn
from plotting import plot_params, plot_io_with_truth

import os
import datetime
import argparse

from zodiax.experimental import deserialise
import jax
from jax import numpy as np, random as jr

import webbpsf
from amigo.core import AMIOptics, SUB80Ramp, BaseModeller
from amigo.fitting import optimise
import optax

from matplotlib import pyplot as plt
import ehtplot
import scienceplots

plt.style.use(['science', 'bright', 'no-latex'])

jax.config.update("jax_enable_x64", True)

"""Argument parsing"""
parser = argparse.ArgumentParser()
parser.add_argument("L1_index", help="index of the job", type=int)
args = parser.parse_args()
L1_index = args.L1_index


"""Changing stuff"""
# optimisation
n_epoch = 200

L1_weights = np.linspace(1, 2, 8)
L1 = L1_weights[L1_index]
# L1 = 0.0e0
print("L1:", L1)

config = {
    # # Crude solver
    "position": optax.adam(delay(2e-3, 0), b1=0.7),
    "log_flux": optax.adam(delay(5e-3, 0), b1=0.7),
    # # "position": optax.adam(delay(5e-4, 0), b1=0.7),
    # # "log_flux": optax.adam(delay(2e-3, 0), b1=0.7),
    "volc_frac": optax.adam(delay(2e-3, 0), b1=0.7),
    "log_volcanoes": optax.adam(delay(5e-2, 10), b1=0.7),
    # "distribution": optax.adam(delay(5e-6, 50), b1=0.7),
    # "optics.coefficients": opt(1e3, 10),
    # "one_on_fs": opt(2e4, 15),
}

# outputting
output_dir = "/scratch/user/uqmchar4/code/jwst-io/bunya/outputs/"

# model
ngroups = 3
model_dir = "/scratch/user/uqmchar4/data/jwst/bfe/"
filter_dir = "/scratch/user/uqmchar4/data/jwst/niriss_filters/"

"""Saving stuff"""


def save_script_to_txt(save_dir: str):
    # Get the path of the current Python script
    script_path = os.path.abspath(__file__)

    # Open the script file for reading
    with open(script_path, "r") as script_file:
        script_content = script_file.read()

    # Define the path for the output text file (e.g., 'script.txt')
    output_file_path = save_dir + "script.txt"

    # Write the script content to the output text file
    with open(output_file_path, "w") as output_file:
        output_file.write(script_content)


# now = lambda: datetime.datetime.now().strftime("%H.%M.%S_%d:%m:%Y")
# save_dir = output_dir + now() + "/"
save_dir = output_dir + f"L1_{L1:.2f}/"

# creating save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_script_to_txt(save_dir)  # saving copy of script to save directory

"""Load data."""
io = DefinitelyRealIo(night_day_ratio=1, volc_contrast=1e-2, seed=2, n_volcanoes=3)

"""Set up the model."""
BFE = deserialise(model_dir + "PolyBFE_trained.zdx")
pupil_mask = deserialise(model_dir + "PolyBFE_AMI_mask.zdx")

key = jr.PRNGKey(0)
ngroups = 3
filt = "F430M"
eps = 1e-16

# Get webbpsf optical system for OPD
inst = webbpsf.NIRISS()
# inst.load_wss_opd_by_date(files[0][0].header["DATE-BEG"], verbose=False)
opd = np.array(inst.get_optical_system().planes[0].opd)

# Get dLux optical system
optics = AMIOptics(
    opd=opd,
    pupil_mask=pupil_mask,
    # radial_orders=[0, 1, 2, 3],
    normalise=True,
)
optics = optics.set("coefficients", 5 * jr.normal(key, optics.coefficients.shape))
detector = SUB80Ramp()

params = {}
params["one_on_fs"] = 10 * jr.normal(key, (ngroups, 80, 2))
params["BFE"] = BFE
# params["aberrations"] = np.zeros(optics.coefficients.shape)
# params["biases"] = np.zeros((80, 80))
# params["BFE"] = PolyBFE(ksize=5, oversample=1, orders=[1, 2])
params["optics"] = optics
params["detector"] = detector
params["source"] = ComplexIoSource(
    position=jr.uniform(key, (2,), minval=-0.5, maxval=0.5),
    log_flux=jr.uniform(key, (1,), minval=6.9, maxval=7.1),
    volc_frac=1e-1,
    log_volcanoes=np.log10(io.volcanoes / io.volcanoes.sum() + eps),
    # distribution=canon_io.data,
    spectrum=get_filter_spectrum(
        filt,
        file_path=filter_dir,
    ),
)

true_model = BaseModeller(params)

"""Simulating data"""
clean_int = model_fn(true_model, ngroups=ngroups)
data = add_noise_to_ramp(clean_int)  # adding photon noise to ramp
io_mask = np.where(initialise_disk()==0, False, True)

"""Optimising"""
initial_model = true_model.set(
    [
        "position",
        "log_flux",
        "log_volcanoes",
        "volc_frac",
        # "distribution",
        "optics.coefficients",
        "one_on_fs",
    ],
    [
        true_model.position + 0.1 * jr.normal(jr.PRNGKey(1), shape=(2,)),
        true_model.log_flux - 0.1,
        np.log10(initialise_disk(normalise=True)),
        # true_model.volcanoes,
        0.3,
        # np.zeros_like(true_model.optics.coefficients),
        true_model.optics.coefficients,
        # np.zeros_like(true_model.one_on_fs),
        true_model.one_on_fs,
    ],
)

final_model, losses, params_out, opt_state = optimise(
    initial_model,
    # data,
    {"model_fn": model_fn, "ngroups": ngroups, "data": data, "mask": io_mask, "L1": L1, "ngroups": ngroups},
    loss_fn,
    n_epoch,
    config,
    norm_fn=complex_norm_fn,
    # grad_fn=grad_fn,
    print_grads=False,
    return_state=True,
    nan_method="none",
)

print(save_dir)

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
    # roll_angle_degrees=roll_angle_degrees,
    # cmap="inferno",
    true_model=true_model,
    save=save_dir,
    )

print("0")