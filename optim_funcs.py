from jax import numpy as np, scipy as jsp
import optax


def bias_prior(model, mean=80, std=50):
    # Estimated std is ~25, so we use 50 here to be conservative
    # Estimated mean is taken from the initial bias estimation
    bias_vec = model.biases
    return jsp.stats.norm.logpdf(bias_vec, loc=mean, scale=std).sum()


def L1_loss(model):
    # only applied to the volcano array
    return np.nansum(model.source.volc_frac * np.abs(10**model.source.log_volcanoes))


def L2_loss(model):
    # TODO BRUH THIS NORMALISED
    return np.nansum((model.distribution - model.distribution.mean())**2)


def TV_loss(model):
    array = np.pad(model.distribution, 2)
    diff_y = np.abs(array[1:, :] - array[:-1, :]).sum()
    diff_x = np.abs(array[:, 1:] - array[:, :-1]).sum()
    # return np.hypot(diff_x, diff_y)
    return diff_x + diff_y


def QV_loss(model):
    array = np.pad(model.distribution, 2)
    diff_y = np.square(array[1:, :] - array[:-1, :]).sum()
    diff_x = np.square(array[:, 1:] - array[:, :-1]).sum()
    return diff_x + diff_y


def ME_loss(model, eps=1e-16):
    """
    Maximum Entropy loss function.
    """
    P = model.distribution / np.nansum(model.distribution)
    S = np.nansum(-P * np.log(P + eps))
    return -S


def posterior(
    model, exposure, model_fn, per_pix=True, return_vec=False, return_im=False, **kwargs
):
    # Get the model
    slopes = model_fn(model, exposure, **kwargs)

    # Return vector
    if return_vec:
        return exposure.log_likelihood(slopes)

    # return image
    if return_im:
        return exposure.log_likelihood(slopes, return_im=True)

    # Return mean or sum
    posterior = exposure.log_likelihood(slopes)
    if per_pix:
        return np.nanmean(posterior)
    return np.nansum(posterior)


def loss_fn(model, args, **kwargs):
    posteriors = [
        posterior(
            model, exposure=exp, model_fn=args["model_fn"], per_pix=True, **kwargs
        )
        for exp in args["exposures"]
    ]
    loss = -np.array(posteriors).sum()

    # regularisation
    for reg in args["reg_dict"].keys():
        loss += args["reg_dict"][reg] * args["reg_func_dict"][reg](model)

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

    dist = 10**model.log_distribution
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


def grad_fn(model, grads, args, optimiser):
    for step_mapper in args["step_mappers"]:
        grads = step_mapper.apply(grads)
    return grads


def scheduler(lr, start, *args):
    shed_dict = {start: 1e100}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e100, shed_dict)


base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)
base_adam = lambda vals, **kwargs: optax.adam(vals, **kwargs)

sgd = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))
adam = lambda lr, start, *schedule, **kwargs: base_adam(
    scheduler(lr, start, *schedule), **kwargs
)
