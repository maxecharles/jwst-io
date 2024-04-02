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
    volc_contrast = np.maximum(model.volc_contrast, 0.)
    # distribution = np.maximum(model.distribution, 0.0)
    distribution = np.maximum(model.volcanoes, 0.0)    
    # distribution = model.volcanoes  

    # applying mask
    distribution = np.where(args['mask'], distribution, args['mask'])
    if distribution.sum != 0:
        distribution /= distribution.sum()

    return model.set(
        [
            "spectrum",
            "volcanoes",
            "volc_contrast",
            ],
        [
            spectrum,
            distribution,
            volc_contrast,
            ],
        )

# def grad_fn(grads, args={}, optimisers={}):
#     return grads.set('distribution', np.where(args['mask'], grads, args['mask']))

def L1_loss(model):
    dist = model.get('distribution')
    return np.sqrt(np.nansum(np.abs(dist)))

def loss_fn(model, args={}):
    data = args["data"]
    ramp = args['model_fn'](model)
    loss = np.log10(-jsp.stats.norm.logpdf(ramp, data).sum())
    loss += args["L1"] * L1_loss(model)  # L1
    return loss

delay = lambda lr, s: optax.piecewise_constant_schedule(lr * 1e-16, {s: 1e16})
opt = lambda lr, start: optax.sgd(delay(lr, start), nesterov=True, momentum=0.5)
adam_opt = lambda lr, start: optax.adam(delay(lr, start))
clip = lambda optimiser, v: optax.chain(optimiser, optax.clip(v))