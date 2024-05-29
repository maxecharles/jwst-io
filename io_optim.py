"""Imports"""
from io_funcs import (
    DefinitelyRealIo,
    SimpleIoSource,
    ComplexIoSource,
    get_filter_spectrum,
    add_noise_to_ramp,
    initialise_disk,
    # sim_io_model_fn as model_fn,
    set_all_params,
    grab_first_value,
)
from optim_funcs import sgd, adam, loss_fn, simple_norm_fn, complex_norm_fn, grad_fn
from optim_funcs import L1_loss, L2_loss, TV_loss, ME_loss, QV_loss
from plotting import plot_params, plot_io_with_truth

import os
import datetime
import argparse

from zodiax.experimental import deserialise
import jax
from jax import numpy as np, random as jr

import webbpsf
from amigo.core import AMIOptics, SUB80Ramp, BaseModeller, model_fn
from amigo.fitting import optimise
from amigo.files import get_files, get_exposures, get_filters, initialise_params
from amigo.step_mappers import LocalStepMapper
import optax

from matplotlib import pyplot as plt
import ehtplot
import scienceplots

plt.style.use(["science", "bright", "no-latex"])

jax.config.update("jax_enable_x64", True)
key = jr.PRNGKey(0)


"""Changing stuff"""
# optimisation
n_epoch = 200

coeffs = np.logspace(-3, 2, 15)

config = {
    "positions": sgd(1e-1, 0),
    "fluxes": sgd(3.5e-1, 0),
    "log_distribution": adam(5e-1, 2, (6, 0.3), (30, 0.2), b1=0.7),
    # "volc_frac": adam(5e-2, 5, b1=0.7),
    # "log_volcanoes": adam(5e-1, 2, (6, 0.3), b1=0.7),
    "aberrations": sgd(0*1e-2, 9),
    "one_on_fs": sgd(0*2e-1, 5),
    # "optics.coefficients": opt(1e3, 10),
}

# outputting
output_dir = "/scratch/user/uqmchar4/code/jwst-io/bunya/outputs/"
# output_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/io/output/"

# model
ngroups = 3
model_dir = "/scratch/user/uqmchar4/data/jwst/bfe/"
filter_dir = "/scratch/user/uqmchar4/data/jwst/niriss_filters/"
# model_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/bfe/"
# filter_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/data/niriss_filters/"

# real data
real_data_dir = "/scratch/user/uqmchar4/data/jwst/JWST/ERS1373/calslope_18/"
# real_data_dir = "/Users/mcha5804/JWST/ERS1373/calslope_18/"


# initial model setup
def initialise_model(true_model, exposures, eps=1e-16):
    return true_model.set(
        [
            "positions",
            "fluxes",
            "source.log_distribution",
            # "log_volcanoes",
            # "volc_frac",
            "aberrations",
            "one_on_fs",
        ],
        [
            set_all_params(
                exposures,
                grab_first_value(true_model, "positions")
                + 0.01 * jr.normal(jr.PRNGKey(2), (2,)),
            ),
            set_all_params(exposures, np.array(6.8)),
            np.log10(initialise_disk(normalise=True) + eps),
            # 0.2,
            set_all_params(exposures, grab_first_value(true_model, "aberrations")),
            # true_model.aberrations,
            # set_all_params(exposures, np.zeros_like(grab_first_value(true_model, 'one_on_fs'))),
            set_all_params(exposures, grab_first_value(true_model, "one_on_fs")),
            # true_model.one_on_fs,
        ],
    )


"""Argument parsing"""
parser = argparse.ArgumentParser()
parser.add_argument("reg_index", help="index of the job", type=int)
args = parser.parse_args()
reg_index = args.reg_index

"""Setting up regularisation"""
chunks = [
    [
        {
            reg: float(coeff),
        }
        for coeff in coeffs
    ]
    for reg in ["L2", "TV", "QV", "ME"]
]
reg_dicts = [
    {},
]  # first one is no regularisation
for chunk in chunks:
    for reg_dict in chunk:
        reg_dicts.append(reg_dict)

reg_func_dict = {
    "L1": L1_loss,
    "L2": L2_loss,
    "TV": TV_loss,
    "QV": QV_loss,
    "ME": ME_loss,
}

reg_dict = reg_dicts[reg_index]

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

if len(reg_dict.keys()) == 0:
    save_dir = output_dir + "no_reg/"
else:
    save_dir = output_dir + f"{list(reg_dict.keys())[0]}_{list(reg_dict.values())[0]:.4f}/"

# creating save directory
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

save_script_to_txt(save_dir)  # saving copy of script to save directory

"""Load data and set up model."""

# loading in canonical bfe and pupil mask
BFE = deserialise(model_dir + "PolyBFE_trained.zdx")
pupil_mask = deserialise(model_dir + "PolyBFE_AMI_mask.zdx")


# REAL DATA
file_fn = lambda **kwargs: get_files(
    real_data_dir,
    "calslope",
    EXP_TYPE="NIS_AMI",
    FILTER=["F480M", "F430M", "F380M"],
    **kwargs,
)

files = file_fn()

# OPTICS
nints = 1

# Get webbpsf optical system for OPD
inst = webbpsf.NIRISS()
opd = np.array(inst.get_optical_system().planes[0].opd)

# Get dLux optical system
optics = AMIOptics(
    opd=opd,
    pupil_mask=pupil_mask,
    normalise=True,
)

nsci = 1
sci_files = []

for file in files:
    if file[0].header["TARGPROP"] == "IO":
        sci_files.append(file)
    elif file[0].header["TARGPROP"] == "PSFCAL.2022A-HD2236-K6":
        continue
    else:
        print(f"Unkown target: {file[0].header['TARGPROP']}")
sci_file = sci_files[0]

# Overwriting for simulated data
sci_file[0].header["NGROUPS"] = ngroups
sci_file[0].header["NINTS"] = nints

# temporarily turning this into an exposure so can pass to model_fn
# this is only necessary to tell model_fn what ngroups is
temp_exposures = get_exposures([sci_file], optics)
params = initialise_params(temp_exposures)

params["optics"] = optics

# SOURCE
n_ios = 1

roll_angle_degrees = 26.5  # deg
canon_io = DefinitelyRealIo(night_day_ratio=1, volc_frac=0.2, seed=2, n_volcanoes=3)
filt = "F430M"

params["filters"] = get_filters(files)
wavels, weights = params["filters"][filt]

params["fluxes"] = set_all_params(temp_exposures, np.array(7.0))  # fluxes is log10 flux

params["source"] = SimpleIoSource(
    position=jr.uniform(key, (2,), minval=-0.5, maxval=0.5),
    log_flux=jr.uniform(key, (1,), minval=6.9, maxval=7.1),
    log_distribution=np.log10(canon_io.distribution),
    spectrum=get_filter_spectrum(filt, file_path=filter_dir),
)

# params["source"] = ComplexIoSource(
#     position=jr.uniform(key, (2,), minval=-0.5, maxval=0.5),
#     log_flux=jr.uniform(key, (1,), minval=6.9, maxval=7.1),
#     volc_frac=1e-1,
#     log_volcanoes=np.log10(canon_io.volcanoes / canon_io.volcanoes.sum() + eps),
#     # distribution=canon_io.data,
#     spectrum=get_filter_spectrum(filt, file_path="/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/data/niriss_filters/"),
# )

# DETECTOR
detector = SUB80Ramp()
params["one_on_fs"] = set_all_params(
    temp_exposures, 10 * jr.normal(key, (ngroups, 80, 2))
)
# params["one_on_fs"] = set_all_params(temp_exposures, np.zeros((ngroups, 80, 2)))
params["BFE"] = BFE
# params["BFE"] = PolyBFE(ksize=5, oversample=1, orders=[1, 2])
params["detector"] = detector

# FINAL MODEL
true_model = BaseModeller(params)


"""Simulating data"""
clean_slope = model_fn(true_model, temp_exposures[0])

# defining variance from photon and read noise processes
photon_var = clean_slope / nints
read_noise_var = 100.0 / nints
var = photon_var + read_noise_var  # variances add
std = np.sqrt(var)

# drawing from a normal distribution to get the data
data = jr.normal(key, shape=var.shape) * std + clean_slope

sci_file["SCI"].data = data
sci_file["SCI_VAR"].data = var

sci_files = [sci_file]

# creating exposures
exposures = get_exposures(sci_files, true_model.optics)

"""Step mappers"""
# initial model
initial_model = initialise_model(true_model, exposures)

# Calculate local step matrices
local_mappers = []
for exp in exposures:
    local_mappers.append(LocalStepMapper(initial_model, exp))

args = {
    "model_fn": model_fn,
    "exposures": exposures,
    "step_mappers": local_mappers,
    # "mask": io_mask,
    "reg_dict": reg_dict,
    "reg_func_dict": reg_func_dict,
}

final_model, losses, params_out, opt_state = optimise(
    initial_model,
    args,
    loss_fn,
    n_epoch,
    config,
    norm_fn=simple_norm_fn,
    # norm_fn=complex_norm_fn,
    grad_fn=grad_fn,
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
    exposures,
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
