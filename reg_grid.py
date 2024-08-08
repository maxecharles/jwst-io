import webbpsf
import os
from zodiax.experimental import serialise
import jax
from jax import numpy as np
from optim_funcs import sgd, adam
import argparse
import amigo
from amgio import (
    stats,
    build_model,
)

jax.config.update("jax_enable_x64", True)

###################################################################################################
n_epoch = 80

grid_len = 6
curve = np.linspace(0, 1, grid_len)**4
coeffs = {
    "L2": 2e6 * curve,
    "TV": 2e2 * curve,
    "QV": 1e7 * curve,
    "ME": 1e3 * curve,
}

# output_dir = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/io/output/"
# model_cache = "/Users/mcha5804/Library/CloudStorage/OneDrive-TheUniversityofSydney(Students)/PyCharm/jwst/io/arrays/"
# data_cache = "/Users/mcha5804/JWST/ERS1373/calslope_18/"

output_dir = "/scratch/user/uqmchar4/data/jwst/io_outputs/"
model_cache = "/scratch/user/uqmchar4/data/jwst/arrays/"
data_cache = "/scratch/user/uqmchar4/data/jwst/JWST/ERS1373/calslope_18/"

###################################################################################################

# Bind file path, type and exposure type
file_fn = lambda **kwargs: amigo.files.get_files(
    data_cache,
    "calslope",
    EXP_TYPE="NIS_AMI",
    FILTER=["F480M", "F430M", "F380M"],
    **kwargs,
)

all_files = file_fn()  # loading io and calibrator files

# selecting the IO files out from the calibrator
files = [file for file in all_files if file[0].header["TARGPROP"] == "IO"]

# BUILDING MODEL
model, exposures = build_model.build_io_model(files, model_cache)

# OPTIMISATION
fit_params = [
    "positions",
    "fluxes",
    # "one_on_fs",
    "log_distribution",
    "source_spectrum.coefficients",
]

# calculating fishers
rampless_model = model.set("ramp", None)
fishers = amigo.fisher.calc_fishers(rampless_model, exposures, fit_params)

"""Argument parsing"""
parser = argparse.ArgumentParser()
parser.add_argument("reg_index", help="index of the job", type=int)
args = parser.parse_args()
reg_index = args.reg_index

"""Setting up regularisation"""
chunks = [[{reg: float(coeff)} for coeff in coeffs[reg]] for reg in ["L2", "TV", "QV", "ME"]]
reg_dicts = []
for chunk in chunks:
    for reg_dict in chunk:
        reg_dicts.append(reg_dict)
reg_dict = reg_dicts[reg_index]
reg_key = list(reg_dict.keys())[0] + "_" + f"{list(reg_dict.values())[0]:.4e}"

args = {
    "reg_dict": reg_dict,
    "reg_func_dict": stats.reg_func_dict,
}

config = {
    "positions": sgd(3e-1, 0),
    "fluxes": sgd(2e-1, 3),
    # "one_on_fs": sgd(1e-1, 9),
    "log_distribution": adam(2e-1, 4, (10, 0.25), b1=0.7),
    "source_spectrum.coefficients": sgd(1e-1, 5),
}

# running optimisation
final_model, losses, histories, states = amigo.fitting.optimise(
    model,
    exposures,
    optimisers=config,
    fishers=fishers,
    loss_fn=stats.loss_fn,
    args=args,
    epochs=n_epoch,
    # grad_fn=grad_fn,
    print_grads=False,
)

# saving outputs
fit_params = list(config.keys())
save_dir = output_dir + reg_key + "/"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

serialise(save_dir + "final_params.zdx", final_model.get(fit_params))
np.save(save_dir + "losses.npy", np.array(losses))
serialise(save_dir + "histories.zdx", histories[0])
