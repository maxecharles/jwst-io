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


def L2_loss(model):
    return np.nansum(model.distribution**2)


def TV_loss(model):
    # TODO check if this is right
    # Calculate differences along the x and y axes
    d_x = np.diff(model.distribution, axis=1)
    d_y = np.diff(model.distribution, axis=0)

    # Calculate the sum of absolute differences
    return np.sqrt(d_x**2 + d_y**2)


def ME_loss(model):
    flat = model.distribution.ravel()
    hist, _ = np.histogram(flat, bins=100)
    P = hist / hist.sum()
    S = np.nansum(P * np.log(P))
    return -S


def loss_fn(model, args={}):
    data = args["data"]
    ramp = args["model_fn"](model, ngroups=args["ngroups"])
    loss = np.log10(-jsp.stats.norm.logpdf(ramp, data).sum())

    # looping over regularisation coefficients and functions
    for reg in args['reg_dict'].keys():
        coeff, func = args['reg_dict'][reg], args['reg_func_dict'][reg]
        if coeff is not None and coeff != 0.:
            loss = loss + coeff * func(model)
            
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

    dist = 10 ** model.log_distribution
    dist = dist / dist.sum()
    log_distribution = np.log10(dist)

    # # applying mask
    # distribution = np.where(args['mask'], distribution, args['mask'])
    # if distribution.sum != 0:
    #     distribution /= distribution.sum()

    return model.set(
        [
            "spectrum",
            "log_distribution",
        ],
        [
            spectrum,
            log_distribution,
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
    volc_frac = np.clip(model.volc_frac, 0.0, 1.0)

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


delay = lambda lr, s: optax.piecewise_constant_schedule(lr * 1e-16, {s: 1e16})
opt = lambda lr, start: optax.sgd(delay(lr, start), nesterov=True, momentum=0.5)
adam_opt = lambda lr, start: optax.adam(delay(lr, start))
clip = lambda optimiser, v: optax.chain(optimiser, optax.clip(v))
