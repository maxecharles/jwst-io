from jax import numpy as np, scipy as jsp
import optax


def bias_prior(model, mean=80, std=50):
    # Estimated std is ~25, so we use 50 here to be conservative
    # Estimated mean is taken from the initial bias estimation
    bias_vec = model.biases
    return jsp.stats.norm.logpdf(bias_vec, loc=mean, scale=std).sum()


def norm_fn(model, args={}):
    """
    Method for returning a new source object with a normalised total
    spectrum and source distribution.

    Returns
    -------
    source : Source
        The source object with the normalised spectrum and distribution.
    """
    spectrum = model.spectrum.normalise()
    volc_frac = np.maximum(model.volc_frac, 0.0)
    # distribution = np.maximum(model.distribution, 0.0)
    distribution = np.maximum(model.volcanoes, 0.0)
    # distribution = model.volcanoes

    # applying mask
    distribution = np.where(args["mask"], distribution, args["mask"])
    if distribution.sum != 0:
        distribution /= distribution.sum()

    return model.set(
        [
            "spectrum",
            "volcanoes",
            "volc_frac",
        ],
        [
            spectrum,
            distribution,
            volc_frac,
        ],
    )


# def grad_fn(grads, args={}, optimisers={}):
#     return grads.set('distribution', np.where(args['mask'], grads, args['mask']))



def L1_loss(model):
    return np.nansum(model.source.volc_frac * np.abs(10**model.source.log_volcanoes))

def TV_loss(model):
    # TODO check if this is right
    # Calculate differences along the x and y axes
    d_x = np.diff(model.distribution, axis=1)
    d_y = np.diff(model.distribution, axis=0)
    
    # Calculate the sum of absolute differences
    return np.sum(np.abs(d_x)) + np.sum(np.abs(d_y))

def loss_fn(model, args={}):
    data = args["data"]
    ramp = args["model_fn"](model, ngroups=args["ngroups"])
    loss = np.log10(-jsp.stats.norm.logpdf(ramp, data).sum())
    loss += args["L1"] * L1_loss(model)  # L1
    # loss += args["TV"] * TV_loss(model)  # L1
    return loss


def simple_norm_fn(model, args={}):
    """
    Method for returning a new source object with a normalised total
    spectrum and source distribution.

    Returns
    -------
    source : Source
        The source object with the normalised spectrum and distribution.
    """
    spectrum = model.spectrum.normalise()
    distribution = np.maximum(model.distribution, 0.0)    

    # # applying mask
    # distribution = np.where(args['mask'], distribution, args['mask'])
    # if distribution.sum != 0:
    #     distribution /= distribution.sum()

    return model.set(
        [
            "spectrum",
            "distribution",
            ],
        [
            spectrum,
            distribution,
            ],
        )


def complex_norm_fn(model, args={}):
    """
    Method for returning a new source object with a normalised total
    spectrum and source distribution.

    Returns
    -------
    source : Source
        The source object with the normalised spectrum and distribution.
    """
    spectrum = model.spectrum.normalise()
    volc_frac = np.clip(model.volc_frac, 0., 1.)

    volcanoes = np.power(10, model.log_volcanoes)
    volcanoes /= volcanoes.sum()

    log_volcanoes = np.log10(volcanoes)    

    # # applying mask
    # volcanoes = np.where(args['mask'], volcanoes, args['mask'])
    # if volcanoes.sum != 0:
    #     volcanoes /= volcanoes.sum()

    return model.set(
        [
            "spectrum",
            "volc_frac",
            "log_volcanoes",
            ],
        [
            spectrum,
            volc_frac,
            log_volcanoes,
            ],
        )


def grad_fn(grads, args={}, optimisers={}):
    return grads.set("distribution", np.where(args["mask"], grads, args["mask"]))


def maxent_loss(model):
    pass


delay = lambda lr, s: optax.piecewise_constant_schedule(lr * 1e-16, {s: 1e16})
opt = lambda lr, start: optax.sgd(delay(lr, start), nesterov=True, momentum=0.5)
adam_opt = lambda lr, start: optax.adam(delay(lr, start))
clip = lambda optimiser, v: optax.chain(optimiser, optax.clip(v))
