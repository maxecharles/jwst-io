import webbpsf
from jax import numpy as np
import amigo
from amgio import models, model_fits
from zodiax.experimental import deserialise


def build_io_model(files: list, model_cache: str):

    ramp_coeffs = np.load(model_cache + "ramp_coeffs.npy")

    # Get webbpsf optical system for OPD
    inst = webbpsf.NIRISS()
    inst.load_wss_opd_by_date(files[0][0].header["DATE-BEG"], verbose=False)
    opd = np.array(inst.get_optical_system().planes[0].opd)

    ### OPTICS
    # Get dLux optical system
    optics = amigo.core_models.AMIOptics(
        opd=opd,
        normalise=True,
        )

    ### DETECTOR
    # detector = amigo.detector_models.LinearDetectorModel()
    ramp_model = amigo.ramp_models.PolyNonLin().set("coeffs", ramp_coeffs)
    # read_model = amigo.read_models.ReadModel()

    # Setting up science model
    fit = model_fits.IoResolvedFit()
    exposures = amigo.files.get_exposures(files, fit)
    params = amigo.files.initialise_params(exposures, optics)

    source_size = 100  # in oversampled pixels
    ones = np.ones((source_size, source_size))
    log_distribution = np.log10(ones / ones.sum())

    log_distributions = {}
    for exp in exposures:
        dist_key = exp.get_key("log_distribution")
        log_distributions[dist_key] = log_distribution

    params["log_distribution"] = log_distributions

    initial_model = models.IoAmigoModel(
        files,
        params,
        source_coeffs=[1., 0.,],
        optics=optics,
        ramp=ramp_model,
        )

    ab_dict = deserialise(model_cache + "aberrations.zdx")

    # Setting io aberrations to those recovered from calibrator
    model = initial_model.set("aberrations", ab_dict)

    return model, exposures