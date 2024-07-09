import dLux
from dLux import utils as dlu

from jax import Array, vmap, numpy as np, random as jr, scipy as jsp

import numpy as onp

from io_funcs import blur_distribution
from volcanoes import volcanoes

import planetmapper

from matplotlib import pyplot as plt, colormaps
from matplotlib.transforms import Affine2D
from matplotlib.colors import PowerNorm

inferno = colormaps["inferno"]
viridis = colormaps["viridis"]
seismic = colormaps["seismic"]

inferno.set_bad("k", 0.5)
viridis.set_bad("k", 0.5)
seismic.set_bad("k", 0.5)


def plot_ephemeris(ax, legend=False):
    body = io_on_that_day(n_volc=10)

    body.plot_wireframe_angular(
        ax,
        add_title=False,
        label_poles=True,
        indicate_equator=True,
        indicate_prime_meridian=False,
        grid_interval=15,
        grid_lat_limit=75,
        aspect_adjustable="box",
        formatting={
            "limb": {
                "linestyle": "--",
                "linewidth": 0.8,
                "alpha": 0.8,
                "color": "white",
            },
            "grid": {
                "linestyle": "--",
                "linewidth": 0.5,
                "alpha": 0.8,
                "color": "white",
            },
            "equator": {"linewidth": 1, "color": "r", "label": "equator"},
            "terminator": {
                "linewidth": 1,
                "linestyle": "-",
                "color": "aqua",
                "alpha": 0.7,
                "label": "terminator",
            },
            "coordinate_of_interest_lonlat": {
                "color": "g",
                "marker": "^",
                "s": 50,
                "label": "volcano",
            },
            # 'limb_illuminated': {'color': 'b'},
        },
    )

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
        )


def plotting_io_comparison(
    model,
    initial_distribution,
    save: str = None,
    roll_angle_degrees=0.0,
    cmap="afmhot_10u",
    eph_cmap="gist_gray",
    io_max=None,
    power=0.5,
):
    fin_dist = model.source.distribution

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(12, 3))

    # Plot initial distribution
    im0 = plot_io(
        axs[0],
        initial_distribution,
        roll_angle_degrees=roll_angle_degrees,
        model=model,
        vmax=io_max,
        cmap=cmap,
        power=power,
    )
    fig.colorbar(im0, ax=axs[0], label="flux")
    axs[0].set_title("Initial Distribution")

    im1 = plot_io(
        axs[1],
        fin_dist,
        roll_angle_degrees=roll_angle_degrees,
        model=model,
        vmax=io_max,
        cmap=cmap,
        power=power,
    )

    fig.colorbar(im1, ax=axs[1], label="flux")
    axs[1].set_title("Io Recovered Distribution")

    # Plot initial distribution
    plot_ephemeris(
        axs[2],
    )
    im2 = plot_io(
        axs[2],
        fin_dist,
        roll_angle_degrees=roll_angle_degrees,
        model=model,
        vmax=io_max,
        cmap=eph_cmap,
        power=2,
    )
    fig.colorbar(im2, ax=axs[2], label="flux")
    axs[2].set_title("With Ephemeris")

    plt.tight_layout()
    if save is not None:
        plt.savefig(f"{save}result.pdf")
        plt.close()
    else:
        plt.show()


def format_fn(params_out, param, ax, alpha=0.75, true_model=None):
    if true_model is not None:
        truth = True
    else:
        truth = False

    ax.set(title=param, xlabel="Epochs")

    try:
        arr = np.array(list(prep_observations_for_plot(params_out)[param]))
    except:
        arr = np.array(list(prep_sim_params_for_plot(params_out)[param]))

    if param == "distribution":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr, alpha=0.05, linewidth=1)
        ax.set(ylabel="Source Distribution")
        ax.axhline(0, color="k", linestyle="--")

    if param == "log_distribution":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr, alpha=0.05, linewidth=1)
        ax.set(ylabel="Log Source Distribution")

    elif param == "volcanoes":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr, alpha=0.05, linewidth=1)
        ax.set(ylabel="Volcano Distribution")
        ax.axhline(0, color="k", linestyle="--")

    elif param == "log_volcanoes":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr, alpha=0.05, linewidth=1)
        ax.set(ylabel="Volcano Distribution")

    elif param == "volc_contrast":
        ax.plot(arr)
        ax.set(ylabel="Volcano Contrast")
        # if truth:
        #     ax.axhline(true_model.volc_contrast, color="r", linestyle="--")

    elif param == "positions" or param == "position":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr - arr[0])
        ax.set(ylabel="Position (arcsec)")
        if truth:
            for pos in true_model.position:
                ax.axhline(pos, color="r", linestyle="--")

    elif param == "log_flux" or param == "log_fluxes" or param == "fluxes":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr)
        ax.set(ylabel="Flux (photons)")
        if truth:
            ax.axhline(true_model.log_flux, color="r", linestyle="--")

    elif "aberrations" in param or param == "optics.coefficients":
        arr = arr.reshape(arr.shape[0], -1)
        arr -= arr[0]
        ax.plot(arr - arr[0], alpha=0.25)
        ax.set(ylabel="Aberrations (nm)")

    elif "one_on_fs" in param:
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr - arr[0], alpha=0.25)
        ax.set(ylabel="OneOnFs")

    elif "BFE.linear" in param:
        arr = arr.reshape(len(arr), -1)
        ax.plot(arr - arr[0], alpha=0.5, linewidth=1)
        ax.set(ylabel="BFE Linear")

    elif "BFE.quadratic" in param:
        arr = arr.reshape(len(arr), -1)
        ax.plot(arr - arr[0], alpha=0.5)
        ax.set(ylabel="BFE Quadratic")

    elif "pupil_mask.holes" in param:
        arr = arr.reshape(len(arr), -1)
        arr -= arr[0]
        arr *= 1e3
        ax.plot(arr, alpha=alpha)
        ax.set(ylabel="Pupil Mask Holes (mm)")

    elif "bias" in param:
        arr = arr.reshape(len(arr), -1)
        ax.plot(arr - arr[0], alpha=0.25)
        ax.set(ylabel="Bias")

    elif "PRF" in param:
        ax.plot(arr.reshape(len(arr), -1), alpha=0.25)
        ax.set(ylabel="PRF")

    elif "rotation" in param:
        ax.plot(dlu.rad2deg(arr))
        ax.set(ylabel="Rotation (deg)")

    elif "stars" in param:
        Teffs = []
        for star in leaf.values():
            Teffs.append(np.array([s.Teff for s in star]))
        Teffs = np.array(Teffs).T
        ax.plot(Teffs)
        ax.set(ylabel="Stellar Teff (K)")

    else:
        print(param)
        ax.plot(arr)

    return ax


def plot_params(
    losses,
    params_out,
    format_fn=format_fn,
    k=10,
    l=-1,
    save: str = None,
    true_model=None,
):
    plt.figure(figsize=(9, 3))
    plt.subplot(1, 2, 1)
    plt.title("Full Loss")
    plt.plot(losses)

    if k >= len(losses):
        k = 0
    last_losses = losses[k:l]
    n = len(last_losses)
    plt.subplot(1, 2, 2)
    plt.title(f"Final {n} Losses")
    plt.plot(np.arange(k, k + n), last_losses)

    plt.tight_layout()
    if save is not None:
        plt.savefig(f"{save}loss.pdf")
        plt.close()
    else:
        plt.show()

    params = list(params_out.params.keys())
    for i in np.arange(0, len(params), 2):
        fig = plt.figure(figsize=(9, 3))
        ax = plt.subplot(1, 2, 1)
        # plt.title(params[i])
        ax = format_fn(params_out, params[i], ax=ax, true_model=true_model)
        # plt.plot(format_fn(params_out, params[i]))

        ax = plt.subplot(1, 2, 2)
        if i + 1 == len(params):
            plt.tight_layout()
            plt.show()
            break
        # plt.title(params[i + 1])
        # plt.plot(format_fn(params_out, params[i + 1]))
        ax = format_fn(params_out, params[i + 1], ax=ax, true_model=true_model)

        plt.tight_layout()
        if save is not None:
            plt.savefig(f"{save}{params[i]}_{params[i+1]}.pdf")
            plt.close()
        else:
            plt.show()


def prep_observations_for_plot(params_out):
    prepped_params_out = {}
    for p, observations in zip(params_out.params.keys(), params_out.params.values()):
        obs_list = []
        if type(observations) == dict:
            for obs in observations.values():
                obs = np.array(obs)
                if len(obs.shape) < 2:
                    obs = obs.reshape(-1, 1)
                # print(obs_array.shape)
                obs_list.append(obs)
            prepped_params_out[p] = np.hstack(np.array([*obs_list]))

        elif type(observations) == list:
            obs = np.array(observations)
            if len(obs.shape) < 2:
                obs = obs.reshape(-1, 1)
            prepped_params_out[p] = obs

    return prepped_params_out


def prep_sim_params_for_plot(params_out):
    prepped_params_out = {}
    for param, value in zip(params_out.params.keys(), params_out.params.values()):
        value = np.array(value)

        if len(value.shape) > 2:
            value = value.reshape(value.shape[0], -1)

        prepped_params_out[param] = value

    return prepped_params_out


def plotting_data_comparison(model, exposures, model_fn, power=0.5):
    nrows = 2 * len(exposures)

    plt.figure(figsize=(15, 3 * nrows))

    for exp_idx, exp in enumerate(exposures):
        model_imgs = model_fn(model, exp)

        for grp_idx, grp_no in enumerate(np.array([0, -1])):
            plt.subplot(nrows, 4, 1 + 4 * grp_idx + 8 * exp_idx)
            plt.imshow(exp.data[grp_no], cmap=inferno, norm=PowerNorm(power, vmin=0))
            plt.colorbar(label=None)
            plt.title(f"Data. Exp:{exp_idx}, Grp:{grp_no}")
            plt.xticks([0, model_imgs[grp_no].shape[0] - 1])
            plt.yticks([0, model_imgs[grp_no].shape[1] - 1])

            plt.subplot(nrows, 4, 2 + 4 * grp_idx + 8 * exp_idx)
            plt.imshow(model_imgs[grp_no], cmap=inferno, norm=PowerNorm(power, vmin=0))
            plt.colorbar()
            plt.title(f"Model Image. Exp:{exp_idx}, Grp:{grp_no}")
            plt.xticks([0, model_imgs[grp_no].shape[0] - 1])
            plt.yticks([0, model_imgs[grp_no].shape[1] - 1])

            residuals, bound_dict = get_residuals(
                exp.data[grp_no], model_imgs[grp_no], return_bounds=True
            )
            plt.subplot(nrows, 4, 3 + 4 * grp_idx + 8 * exp_idx)
            plt.imshow(residuals, **bound_dict)
            plt.colorbar(label="flux")
            plt.title(f"Residuals. Exp:{exp_idx}, Grp:{grp_no}")
            plt.xticks([0, model_imgs[grp_no].shape[0] - 1])
            plt.yticks([0, model_imgs[grp_no].shape[1] - 1])

            nnim = residuals / np.sqrt(exp.variance[grp_no])
            bound_dict = get_residual_bounds(nnim)
            plt.subplot(nrows, 4, 4 + 4 * grp_idx + 8 * exp_idx)
            plt.imshow(nnim, **bound_dict)
            plt.colorbar()
            plt.title(f"Noise Normalised Residuals")
            plt.xticks([0, model_imgs[grp_no].shape[0] - 1])
            plt.yticks([0, model_imgs[grp_no].shape[1] - 1])

    plt.show()


def get_arcsec_extents(pixel_scale, shape):
    """
    Get the arcsec extents of an image given the pixel scale and shape.
    """
    return np.array([-0.5, 0.5, -0.5, 0.5]) * pixel_scale * shape[0]


def plot_io(
    ax,
    array,
    roll_angle_degrees: float = 0.0,
    pixel_scale: float = 0.0656 / 4,
    model=None,
    show_diff_lim: bool = True,
    cmap: str = "afmhot_10u",
    bg_color: str = "k",
    axis_labels: dict = {
        "xlabel": r"$\Delta$RA [arcsec]",
        "ylabel": r"$\Delta$DEC [arcsec]",
    },
    vmin: float = 0.0,
    vmax: float = None,
    power=0.5,
):
    rotation_transform = Affine2D().rotate_deg(
        roll_angle_degrees
    )  # Create a rotation transformation

    ax.set_facecolor(bg_color)  # Set the background colour to black
    ax.set(**axis_labels)  # Set the axis labels
    if model is not None:
        pixel_scale = model.psf_pixel_scale / model.optics.oversample
        if show_diff_lim:
            ax = plot_diffraction_limit(model, ax, OOP=True)
    im = ax.imshow(
        array,
        cmap=cmap,
        extent=get_arcsec_extents(pixel_scale, array.shape),
        norm=PowerNorm(power, vmin=vmin, vmax=vmax),
    )

    trans_data = rotation_transform + ax.transData  # creating transformation
    im.set_transform(trans_data)  # applying transformation to image

    return im


def plot_io_with_truth(
    model,
    model_fn,
    exposures,
    losses,
    ngroups,
    opt_state,
    initial_distribution=None,
    true_model=None,
    save: str = None,
    roll_angle_degrees=0.0,
    cmap="afmhot_10u",
    io_max=None,
):
    fin_dist = model.source.distribution

    nrows = 4

    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(17, 3 * nrows + 5))

    im0 = plot_io(
        axs[0, 0],
        fin_dist,
        roll_angle_degrees=roll_angle_degrees,
        model=model,
        vmax=io_max,
        cmap=cmap,
    )
    fig.colorbar(im0, ax=axs[0, 0], label="flux")
    axs[0, 0].set_title("Io Recovered Distribution")

    blurred = blur_distribution(true_model)
    im1 = plot_io(
        axs[0, 1],
        blurred,
        roll_angle_degrees=roll_angle_degrees,
        model=true_model,
        vmax=io_max,
        cmap=cmap,
    )
    fig.colorbar(im1, ax=axs[0, 1], label="flux")
    axs[0, 1].set_title("Blurred Truth Distribution")

    # Plot true distribution
    im2 = plot_io(
        axs[0, 2],
        true_model.distribution,
        roll_angle_degrees=roll_angle_degrees,
        model=true_model,
        vmax=io_max,
        cmap=cmap,
    )
    fig.colorbar(im2, ax=axs[0, 2], label="flux")
    axs[0, 2].set_title(f"True Distribution")

    # Plot initial distribution
    im3 = plot_io(
        axs[1, 0],
        initial_distribution,
        roll_angle_degrees=roll_angle_degrees,
        model=model,
        vmax=io_max,
        cmap=cmap,
    )
    fig.colorbar(im3, ax=axs[1, 0], label="flux")
    axs[1, 0].set_title("Initial Distribution")

    # Plot final residuals
    io_resids, bound_dict = get_residuals(fin_dist, blurred, return_bounds=True)
    im4 = plot_io(
        axs[1, 1],
        io_resids,
        roll_angle_degrees=roll_angle_degrees,
        model=model,
        **bound_dict,
        show_diff_lim=False,
        bg_color="white",
        power=1.0,
    )
    fig.colorbar(im4, ax=axs[1, 1], label="flux")
    axs[1, 1].set_title(f"Comparison to Blurred Truth")

    # Plot final residuals
    io_resids, bound_dict = get_residuals(
        fin_dist, true_model.distribution, return_bounds=True
    )
    im5 = plot_io(
        axs[1, 2],
        io_resids,
        roll_angle_degrees=roll_angle_degrees,
        model=model,
        **bound_dict,
        show_diff_lim=False,
        bg_color="white",
        power=1.0,
    )
    fig.colorbar(im5, ax=axs[1, 2], label="flux")
    axs[1, 2].set_title(f"Final Residuals. Loss: {losses[-1]:.1f}")

    model_imgs = model_fn(model, exposures[0])

    for grp_idx, grp_no in enumerate(np.arange(-1, 1)):
        im5 = axs[2 + grp_idx, 0].imshow(
            model_imgs[grp_no],
            cmap="cividis",
            vmin=0,
        )
        fig.colorbar(im5, ax=axs[2 + grp_idx, 0], label="flux")
        axs[2 + grp_idx, 0].set_title(f"Model Image, Grp:{grp_no}")
        axs[2 + grp_idx, 0].set_xticks([0, fin_dist.shape[0] - 1])
        axs[2 + grp_idx, 0].set_yticks([0, fin_dist.shape[1] - 1])

        im6 = axs[2 + grp_idx, 1].imshow(
            exposures[0].data[grp_no],
            cmap="cividis",
            vmin=0,
        )
        fig.colorbar(im6, ax=axs[2 + grp_idx, 1], label="flux")
        axs[2 + grp_idx, 1].set_title(f"Data, Grp:{grp_no}")
        axs[2 + grp_idx, 1].set_xticks([0, fin_dist.shape[0] - 1])
        axs[2 + grp_idx, 1].set_yticks([0, fin_dist.shape[1] - 1])

        residuals, bound_dict = get_residuals(
            model_imgs[grp_no], exposures[0].data[grp_no], return_bounds=True
        )
        im7 = axs[2 + grp_idx, 2].imshow(residuals, **bound_dict)
        fig.colorbar(im7, ax=axs[2 + grp_idx, 2], label="flux")
        axs[2 + grp_idx, 2].set_title(f"Residuals, Grp:{grp_no}")
        axs[2 + grp_idx, 2].set_xticks([0, fin_dist.shape[0] - 1])
        axs[2 + grp_idx, 2].set_yticks([0, fin_dist.shape[1] - 1])

    if save is not None:
        plt.savefig(f"{save}result.pdf")
        plt.close()
    else:
        plt.show()


def plot_diffraction_limit(model, ax=None, OOP=False):
    diff_lim = dlu.rad2arcsec(model.source.wavelengths.mean() / model.optics.diameter)
    scale_length = diff_lim

    scale_bar_x = -0.55
    scale_bar_y = -0.55

    if OOP and ax is not None:
        ax.plot(
            [scale_bar_x, scale_bar_x + scale_length],
            [scale_bar_y, scale_bar_y],
            color="hotpink",
            linewidth=2,
        )
        ax.text(
            scale_bar_x + scale_length / 2 - 0.046,
            scale_bar_y + 0.02,
            r"$\lambda / D$",
            color="hotpink",
            fontsize=8,
        )
        return ax

    else:
        plt.plot(
            [scale_bar_x, scale_bar_x + scale_length],
            [scale_bar_y, scale_bar_y],
            color="hotpink",
            linewidth=2,
        )
        plt.text(
            scale_bar_x + scale_length / 2 - 0.046,
            scale_bar_y + 0.02,
            r"$\lambda / D$",
            color="hotpink",
            fontsize=8,
        )


def io_on_that_day(
    n_volc="all", body="io", date="2022-08-01T16:52:00.000", observer="jwst", **kwargs
):
    """
    Use this with body.plot_wireframe_angular to plot the body on a map.
    """
    body = planetmapper.Body(body, date, observer=observer, **kwargs)

    if n_volc == "all":
        n_volc = len(volcanoes)
    body.coordinates_of_interest_lonlat = volcanoes[:n_volc]

    return body


def plot_io_with_ephemeris(
    ax, array, roll_angle_degrees=246.80584209034947, legend=False, **kwargs
):
    body = io_on_that_day(n_volc=10)

    plot_io(ax, array, roll_angle_degrees, show_diff_lim=True, **kwargs)

    body.plot_wireframe_angular(
        ax,
        add_title=False,
        label_poles=True,
        indicate_equator=True,
        indicate_prime_meridian=False,
        grid_interval=15,
        grid_lat_limit=75,
        aspect_adjustable="box",
        formatting={
            "limb": {
                "linestyle": "--",
                "linewidth": 0.8,
                "alpha": 0.8,
                "color": "white",
            },
            "grid": {
                "linestyle": "--",
                "linewidth": 0.5,
                "alpha": 0.8,
                "color": "white",
            },
            "equator": {"linewidth": 1, "color": "r", "label": "equator"},
            "terminator": {
                "linewidth": 1,
                "linestyle": "-",
                "color": "aqua",
                "alpha": 0.7,
                "label": "terminator",
            },
            "coordinate_of_interest_lonlat": {
                "color": "g",
                "marker": "^",
                "s": 50,
                "label": "volcano",
            },
            # 'limb_illuminated': {'color': 'b'},
        },
    )

    if legend:
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(
            by_label.values(),
            by_label.keys(),
            loc="upper left",
        )


def get_residual_bounds(arr: Array):
    extent = np.nanmax(np.abs(arr))
    bound_dict = {"vmin": -extent, "vmax": extent, "cmap": seismic}
    return bound_dict


def get_residuals(arr1: Array, arr2: Array, return_bounds: bool = False):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    residuals = arr1 - arr2

    if return_bounds:
        bound_dict = get_residual_bounds(residuals)
        return residuals, bound_dict

    return residuals


def get_extents(data, pixel_scale: float = None):
    x = data.shape[0]
    if pixel_scale is not None:
        x *= pixel_scale
    return {"extent": (-x / 2, x / 2, -x / 2, x / 2)}
