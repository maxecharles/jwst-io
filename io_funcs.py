import os

import dLux
from dLux.sources import Source
from dLux.psfs import PSF
from dLux import utils as dlu

from jax import Array, vmap, numpy as np, random as jr, scipy as jsp
from jax.scipy.signal import convolve

from astropy import units as u
import datetime

import numpy as onp
from numpy import random as rd

from matplotlib import pyplot as plt
from matplotlib import colormaps
from matplotlib.transforms import Affine2D

seismic = colormaps["seismic"]
seismic.set_bad("k", 0.5)

# plt.rcParams["font.size"] = 7

Optics = lambda: dLux.optical_systems.BaseOpticalSystem
Spectrum = lambda: dLux.spectra.BaseSpectrum


class IoSource(Source):
    """
    A single resolved source with a spectrum, position, flux, and distribution array
    that represents the resolved component.

    ??? abstract "UML"
        ![UML](../../assets/uml/ResolvedSource.png)

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : float, photons
        The flux of the object.
    distribution : Array
        The array of intensities representing the resolved source.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    distribution: Array
    log_flux: Array
    position: Array

    def __init__(
        self: Source,
        wavelengths: Array = None,
        position: Array | tuple = np.zeros(2),  # arcseconds
        log_flux: Array | float = 7.0,
        distribution: Array = np.ones((3, 3)),
        weights: Array = None,
        spectrum: Spectrum() = None,
    ):
        """
        Parameters
        ----------
        wavelengths : Array, metres
            The array of wavelengths at which the spectrum is defined.
        position : Array, radians = np.zeros(2)
            The (x, y) on-sky position of this object.
        flux : float, photons = 1.
            The flux of the object.
        distribution : Array = np.ones((3, 3))
            The array of intensities representing the resolved source.
        weights : Array = None
            The spectral weights of the object.
        spectrum : Spectrum = None
            The spectrum of this object, represented by a Spectrum object.
        """
        distribution = np.asarray(distribution, dtype=float)
        self.distribution = distribution / distribution.sum()

        self.position = np.array(position, dtype=float)  # arcseconds
        self.log_flux = np.array(log_flux, dtype=float)

        if self.distribution.ndim != 2:
            raise ValueError("distribution must be a 2d array.")

        super().__init__(wavelengths=wavelengths, weights=weights, spectrum=spectrum)

    def model(
        self: Source,
        optics: Optics = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Models the source object through the provided optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source object.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        if return_wf and return_psf:
            raise ValueError(
                "return_wf and return_psf cannot both be True. " "Please choose one."
            )

        weights = self.weights * (10**self.log_flux)

        # Note we always return wf here so we can convolve each wavelength
        # individually if a chromatic wavefront output is required.
        wf = optics.propagate(
            self.wavelengths, dlu.arcsec2rad(self.position), weights, return_wf=True
        )

        # Returning wf is a special case
        if return_wf:
            conv_fn = lambda psf: convolve(psf, self.distribution, mode="same")
            return wf.set("amplitude", vmap(conv_fn)(wf.psf) ** 0.5)

        # Return psf object
        conv_psf = convolve(wf.psf.sum(0), self.distribution, mode="same")
        if return_psf:
            return PSF(conv_psf, wf.pixel_scale.mean())

        # Return array psf
        return conv_psf


class HD2236(Source):
    """
    A single resolved source with a spectrum, position, flux, and distribution array
    that represents the resolved component.

    ??? abstract "UML"
        ![UML](../../assets/uml/ResolvedSource.png)

    Attributes
    ----------
    position : Array, radians
        The (x, y) on-sky position of this object.
    flux : float, photons
        The flux of the object.
    distribution : Array
        The array of intensities representing the resolved source.
    spectrum : Spectrum
        The spectrum of this object, represented by a Spectrum object.
    """

    log_flux: Array
    position: Array

    def __init__(
        self: Source,
        wavelengths: Array = None,
        position: Array | tuple = np.zeros(2),  # arcseconds
        log_flux: Array | float = 5.8389,
        weights: Array = None,
        spectrum: Spectrum() = None,
    ):
        self.position = np.array(position, dtype=float)  # arcseconds
        self.log_flux = np.array(log_flux, dtype=float)

        if self.position.shape != (2,):
            raise ValueError("position must be a 1d array of shape (2,).")

        super().__init__(wavelengths=wavelengths, weights=weights, spectrum=spectrum)

    def model(
        self: Source,
        optics: Optics,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
        """
        Models the source object through the provided optics.

        Parameters
        ----------
        optics : Optics
            The optics through which to model the source object.
        return_wf : bool = False
            Should the Wavefront object be returned instead of the psf Array?
        return_psf : bool = False
            Should the PSF object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront, PSF
            if `return_wf` is False and `return_psf` is False, returns the psf Array.
            if `return_wf` is True and `return_psf` is False, returns the Wavefront
                object.
            if `return_wf` is False and `return_psf` is True, returns the PSF object.
        """
        self = self.normalise()
        weights = self.weights * (10**self.log_flux)
        return optics.propagate(
            self.wavelengths,
            dlu.arcsec2rad(self.position),
            weights,
            return_wf,
            return_psf,
        )


def initialise_disk(pixel_scale=0.065524085, oversample=4, normalise=False):
    io_initial_distance = 4.36097781166671 * u.AU
    io_final_distance = 4.36088492436330 * u.AU
    io_diameter = 3643.2 * u.km  # from wikipedia

    io_mean_distance = (io_initial_distance + io_final_distance).to(u.km) / 2
    angular_size = dlu.rad2arcsec(
        io_diameter / io_mean_distance
    )  # angular size in arcseconds

    npix = oversample * np.ceil(angular_size / pixel_scale).astype(int)
    coords = dlu.pixel_coords(npixels=npix, diameter=npix * pixel_scale / oversample)

    circle = dlu.soft_circle(coords, radius=angular_size / 2, clip_dist=2e-3)

    if not normalise:
        return circle
    return circle / circle.sum()


class DefinitelyRealIo():
    def __init__(
            self,
            n_volcanoes=None,
            night_day_ratio=None,
            terminator_angle=None,  # degrees
            darkness=None,
            R=1.1519 / 2,
            R_volc=1e-2,
            volc_contrast=1.,
            pixel_scale=0.065524085,
            oversample=4,
            seed=None,
    ):  
        
        if seed is not None:
            rd.seed(seed)
        self.seed = seed

        if night_day_ratio is None:
            night_day_ratio = rd.uniform(0.5, 1)
        elif night_day_ratio < 0.5 or night_day_ratio > 1:
            raise ValueError("night_day_ratio must be between 0.5 and 1")

        self.night_day_ratio = night_day_ratio

        if terminator_angle is None:
            terminator_angle = rd.uniform(-180, 180)
        self.terminator_angle = terminator_angle

        if darkness is None:
            darkness = rd.uniform(0.2, 0.8)
        self.darkness = darkness
        
        self.R = R
        self.R_volc = R_volc
        self.pixel_scale = pixel_scale
        self.oversample = oversample
        self.volc_contrast = volc_contrast

        if n_volcanoes is None:
            n_volcanoes = rd.randint(1, 10)
        self.n_volcanoes = n_volcanoes
        self.volcanoes = self.generate_volcanoes()

        self.disk = dLux.PSF(initialise_disk(pixel_scale, oversample, normalise=True), pixel_scale)

        ill_mask = self.generate_illumination_mask(night_day_ratio, terminator_angle, self.disk.data.shape[0])
        self.ill_mask = ill_mask
        

    def generate_illumination_mask(self, axis_ratio, rotation_angle, size):

        semi_major = self.R * self.oversample / self.pixel_scale
        semi_minor = axis_ratio * semi_major

        # Create an empty boolean array
        array = onp.zeros((size, size), dtype=bool)

        # Calculate the center of the array
        center = (size - 1) / 2

        # Iterate over each point in the array
        for i in range(size):
            for j in range(size):
                # Calculate the distance from the center of the array to the current point
                x = i - center
                y = j - center

                # Apply rotation to the coordinates
                x_rot = x * onp.cos(onp.radians(rotation_angle)) - y * onp.sin(onp.radians(rotation_angle))
                y_rot = x * onp.sin(onp.radians(rotation_angle)) + y * onp.cos(onp.radians(rotation_angle))

                # Check if the point falls within the ellipse
                if x_rot**2 + y_rot**2 <= semi_major**2:
                    if onp.arctan2(y_rot, x_rot) > 0:
                        if (x_rot / semi_major)**2 + (y_rot / semi_minor)**2 >= 1:
                        
                            array[i, j] = True
                    elif onp.arctan2(y_rot, x_rot) < 0:
                        array[i, j] = False

        return array
        

    def generate_volcanoes(self):

        disk = dLux.PSF(initialise_disk(self.pixel_scale, self.oversample), self.pixel_scale)

        # generating cartesian coordinates of volcanoes
        volcano_coords = np.array([np.power(rd.uniform(size=(self.n_volcanoes,)), p) for p in [0.5, 1]]).T
        volcano_coords *= np.array(self.n_volcanoes*[[self.R, 2*np.pi],])
        volcano_coords = dlu.polar2cart(volcano_coords.T).T

        # volcano brightness
        volcano_brights = onp.abs(rd.normal(loc=1., scale=0.3, size=(self.n_volcanoes,)))
        volcano_brights = np.array([np.ones_like(disk.data) * bright for bright in volcano_brights])

        # generating coordinate grids for each volcano
        coords = dlu.pixel_coords(npixels=disk.data.shape[0], diameter=disk.data.shape[0] * self.pixel_scale / self.oversample)
        coords = [dlu.translate_coords(coords, volcano_coords[i]) for i in range(self.n_volcanoes)]

        # creating an array for each volcano and summing to have an array with all volcanoes
        volcanoes = np.multiply(np.array([dlu.soft_circle(coord, self.R_volc, clip_dist=1e-2) for coord in coords]), volcano_brights).sum(0)

        return volcanoes / volcanoes.sum()

    @property
    def data(self):
        return np.where(self.ill_mask, self.disk.data * self.darkness, self.disk.data) + self.volc_contrast * self.volcanoes


def get_filter_spectrum(filt: str, file_path: str, nwavels: int = 9):

    if filt not in ["F380M", "F430M", "F480M", "F277W"]:
        raise ValueError("Supported filters are F380M, F430M, F480M, F277W.")

    # filter_path = os.path.join()
    wl_array, throughput_array = np.array(onp.loadtxt(file_path + f"JWST_NIRISS.{filt}.dat", unpack=True))

    edges = np.linspace(wl_array.min(), wl_array.max(), nwavels + 1)
    wavels = np.linspace(wl_array.min(), wl_array.max(), 2 * nwavels + 1)[1::2]

    areas = []
    for i in range(nwavels):
        cond1 = edges[i] < wl_array
        cond2 = wl_array < edges[i + 1]
        throughput = np.where(cond1 & cond2, throughput_array, 0)
        areas.append(jsp.integrate.trapezoid(y=throughput, x=wl_array))

    areas = np.array(areas)
    weights = areas / areas.sum()

    wavels *= 1e-10

    return dLux.Spectrum(wavels, weights)


def plotting_io_comparison(model, model_fn, opt_state, exposures, losses, initial_distribution=None):
    fin_dist = model.source.distribution

    try:
        grads = opt_state[0]['0'].inner_state[0][0].trace.distribution  # TODO this is wrong maybe
    except:
        grads = np.zeros_like(fin_dist)
    nrows = 1 + 2 * len(exposures)

    plt.figure(figsize=(15, 3*nrows))
    plt.subplot(nrows, 4, 1)
    plt.imshow(initial_distribution, cmap="afmhot_10u", vmin=0, vmax=np.nanmax(fin_dist))
    plt.colorbar(label="flux")
    plt.title("Io Initial Distribution")
    plt.xticks([0, fin_dist.shape[0]-1])
    plt.yticks([0, fin_dist.shape[1]-1])

    if np.nanmin(fin_dist) < 0:
        rec_vmin=None
    else:
        rec_vmin=0
    plt.subplot(nrows, 4, 2)
    plt.imshow(fin_dist, cmap="afmhot_10u", vmin=rec_vmin, vmax=np.nanmax(fin_dist))
    plt.colorbar(label="flux")
    plt.title("Io Recovered Distribution")
    plt.xticks([0, fin_dist.shape[0]-1])
    plt.yticks([0, fin_dist.shape[1]-1])

    plt.subplot(nrows, 4, 3)
    plt.imshow(grads, **get_residual_bounds(grads))
    plt.colorbar(label="flux / iter")
    plt.title(f"Final Gradient State. Loss: {losses[-1]:.1f}")
    plt.xticks([0, fin_dist.shape[0]-1])
    plt.yticks([0, fin_dist.shape[1]-1])

    for exp_idx, exp in enumerate(exposures):
        model_imgs = model_fn(model, exp)

        for grp_idx, grp_no in enumerate(np.arange(-1, 1)):
            plt.subplot(nrows, 4, 5 + 4*grp_idx + 8*exp_idx)
            plt.imshow(model_imgs[grp_no], cmap="cividis", vmin=0,)
            plt.colorbar(label="flux")
            plt.title(f"Model Image. Exp:{exp_idx}, Grp:{grp_no}")
            plt.xticks([0, fin_dist.shape[0]-1])
            plt.yticks([0, fin_dist.shape[1]-1])

            plt.subplot(nrows, 4, 6 + 4*grp_idx + 8*exp_idx)
            plt.imshow(exp.data[grp_no], cmap="cividis", vmin=0,)
            plt.colorbar(label="flux")
            plt.title(f"Data. Exp:{exp_idx}, Grp:{grp_no}")
            plt.xticks([0, fin_dist.shape[0]-1])
            plt.yticks([0, fin_dist.shape[1]-1])

            residuals, bound_dict = get_residuals(model_imgs[grp_no], exp.data[grp_no], return_bounds=True)
            plt.subplot(nrows, 4, 7 + 4*grp_idx + 8*exp_idx)
            plt.imshow(residuals, **bound_dict)
            plt.colorbar(label="flux")
            plt.title(f"Residuals. Exp:{exp_idx}, Grp:{grp_no}")
            plt.xticks([0, fin_dist.shape[0]-1])
            plt.yticks([0, fin_dist.shape[1]-1])

            llim = exp.loglike_im(model_imgs)
            plt.subplot(nrows, 4, 8 + 4*grp_idx + 8*exp_idx)
            plt.imshow(-llim)
            plt.colorbar()
            plt.title(f"Neg Log Likelihood")
            plt.xticks([0, model_imgs[grp_no].shape[0]-1])
            plt.yticks([0, model_imgs[grp_no].shape[1]-1])
            
    plt.tight_layout()
    plt.show()


def plot_params(true_model, losses, params_out, format_fn, k=10, l=-1, save: str = None):

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
        ax = format_fn(true_model, params_out, params[i], ax=ax)
        # plt.plot(format_fn(params_out, params[i]))

        ax = plt.subplot(1, 2, 2)
        if i + 1 == len(params):
            plt.tight_layout()
            plt.show()
            break
        # plt.title(params[i + 1])
        # plt.plot(format_fn(params_out, params[i + 1]))
        ax = format_fn(true_model, params_out, params[i + 1], ax=ax)

        plt.tight_layout()
        if save is not None:
            plt.savefig(f"{save}{params[i]}_{params[i+1]}.pdf")
            plt.close()
        else:
            plt.show()


def format_fn(true_model, params_out, param, ax, alpha=0.75):
    ax.set(title=param, xlabel="Epochs")

    leaf = list(params_out.params[param])

    if isinstance(leaf, list):
        arr = np.array(leaf)
    elif param == "stars":
        pass
    else:
        arr = invert_params(leaf)

    if param == "distribution":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr, alpha=0.05, linewidth=1)
        ax.set(ylabel="Source Distribution")
        ax.axhline(0, color='k', linestyle='--')

    elif param == "volcanoes":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr, alpha=0.05, linewidth=1)
        ax.set(ylabel="Volcano Distribution")
        ax.axhline(0, color='k', linestyle='--')

    elif param == "volc_contrast":
        ax.plot(arr)
        ax.set(ylabel="Volcano Contrast")
        ax.axhline(true_model.volc_contrast, color='r', linestyle='--')

    elif param == "positions" or param == "position":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr - arr[0])
        ax.set(ylabel="Position (arcsec)")
        for idx, pos in enumerate(true_model.position):
            ax.axhline(pos - arr[0][idx], color='r', linestyle='--')

    elif param == "log_flux":
        arr = arr.reshape(arr.shape[0], -1)
        ax.plot(arr)
        ax.set(ylabel="Flux (photons)")
        ax.axhline(true_model.log_flux, color='r', linestyle='--')

    elif "aberrations" in param:
        arr = arr.reshape(arr.shape[0], -1)
        arr -= arr[0]
        ax.plot(arr - arr[0], alpha=0.25)
        ax.set(ylabel="Aberrations (nm)")

    elif "OneOnFs" in param:
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
        cmap: str = 'afmhot_10u',
        bg_color: str = 'k',
        axis_labels: dict = {'xlabel': r"$\Delta$RA [arcsec]", 'ylabel': r"$\Delta$DEC [arcsec]"},
        vmin: float = 0.,
        vmax: float = None,
        ):
    

    rotation_transform = Affine2D().rotate_deg(roll_angle_degrees)  # Create a rotation transformation

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
        vmin=vmin,
        vmax=vmax,
        )
    
    trans_data = rotation_transform + ax.transData  # creating transformation
    im.set_transform(trans_data)  # applying transformation to image

    return im


def plot_io_with_truth(model, model_fn, data, losses, ngroups, opt_state, initial_distribution=None, truth=None, save: str = None, roll_angle_degrees=0.0, ):
    fin_dist = model.source.distribution + model.volcanoes

    nrows = 4
    io_max = np.nanmax(truth)

    extents = get_arcsec_extents(model.psf_pixel_scale / model.optics.oversample, fin_dist.shape)

    fig, axs = plt.subplots(nrows=nrows, ncols=3, figsize=(17, 3*nrows + 5))

    # Plot initial distribution
    im0 = plot_io(axs[0, 0], initial_distribution, roll_angle_degrees=roll_angle_degrees, model=model, vmax=io_max)
    fig.colorbar(im0, ax=axs[0, 0], label="flux")
    axs[0, 0].set_title("Io Initial Distribution")

    im1 = plot_io(axs[0, 1], fin_dist, roll_angle_degrees=roll_angle_degrees, model=model, vmax=io_max)
    fig.colorbar(im1, ax=axs[0, 1], label="flux")
    axs[0, 1].set_title("Io Recovered Distribution")


    # Plot true distribution
    im2 = plot_io(axs[0, 2], truth, roll_angle_degrees=roll_angle_degrees, model=model, vmax=io_max)
    fig.colorbar(im2, ax=axs[0, 2], label="flux / iter")
    axs[0, 2].set_title(f"True Distribution")

    # Plot final gradient state
    try:
        grads = opt_state[0]['0'].inner_state[0][0].volcanoes  # TODO this is wrong maybe
        # im3 = axs[1, 0].imshow(grads, **get_residual_bounds(grads), extent=extents)
        im3 = plot_io(axs[1, 0], grads, roll_angle_degrees=roll_angle_degrees, model=model, **get_residual_bounds(grads), show_diff_lim=False, bg_color='white')
        fig.colorbar(im3, ax=axs[1, 0], label="flux / iter")
        axs[1, 0].set_title(f"Final Gradient State. Loss: {losses[-1]:.1f}")
    except:
        print('bruh')

    # Plot final residuals
    io_resids, bound_dict = get_residuals(fin_dist, truth, return_bounds=True)  
    # im4 = axs[1, 1].imshow(io_resids, **bound_dict, extent=extents)
    im4 = plot_io(axs[1, 1], io_resids, roll_angle_degrees=roll_angle_degrees, model=model, **bound_dict, show_diff_lim=False, bg_color='white')
    fig.colorbar(im4, ax=axs[1, 1], label="flux / iter")
    axs[1, 1].set_title(f"Final Residuals. Loss: {losses[-1]:.1f}")

    axs[1, 2].imshow(np.zeros((1, 1)), cmap='Greys')
    axs[1, 2].set(
        xticks=[], yticks=[],
    )

    model_imgs = model_fn(model, n_groups=ngroups)

    for grp_idx, grp_no in enumerate(np.arange(-1, 1)):
        im5 = axs[2+grp_idx, 0].imshow(model_imgs[grp_no], cmap="cividis", vmin=0,)
        fig.colorbar(im5, ax=axs[2+grp_idx, 0], label="flux")
        axs[2+grp_idx, 0].set_title(f"Model Image, Grp:{grp_no}")
        axs[2+grp_idx, 0].set_xticks([0, fin_dist.shape[0]-1])
        axs[2+grp_idx, 0].set_yticks([0, fin_dist.shape[1]-1])

        im6 = axs[2+grp_idx, 1].imshow(data[grp_no], cmap="cividis", vmin=0,)
        fig.colorbar(im6, ax=axs[2+grp_idx, 1], label="flux")
        axs[2+grp_idx, 1].set_title(f"Data, Grp:{grp_no}")
        axs[2+grp_idx, 1].set_xticks([0, fin_dist.shape[0]-1])
        axs[2+grp_idx, 1].set_yticks([0, fin_dist.shape[1]-1])

        residuals, bound_dict = get_residuals(model_imgs[grp_no], data[grp_no], return_bounds=True)
        im7 = axs[2+grp_idx, 2].imshow(residuals, **bound_dict)
        fig.colorbar(im7, ax=axs[2+grp_idx, 2], label="flux")
        axs[2+grp_idx, 2].set_title(f"Residuals, Grp:{grp_no}")
        axs[2+grp_idx, 2].set_xticks([0, fin_dist.shape[0]-1])
        axs[2+grp_idx, 2].set_yticks([0, fin_dist.shape[1]-1])

    if save is not None:
        plt.savefig(f"{save}result.pdf")
        plt.close()
    else:
        plt.show()


def plot_diffraction_limit(model, ax=None, OOP=False):

    pscale = model.optics.psf_pixel_scale / model.optics.oversample
    diff_lim = dlu.rad2arcsec(model.source.wavelengths.mean() / model.optics.diameter)
    scale_length = diff_lim

    scale_bar_x = -0.55
    scale_bar_y = -0.55

    if OOP and ax is not None:
        ax.plot([scale_bar_x, scale_bar_x + scale_length], [scale_bar_y, scale_bar_y], color='hotpink', linewidth=2)
        ax.text(scale_bar_x + scale_length/2 - 0.046, scale_bar_y + 0.02, r'$\lambda / D$', color='hotpink', fontsize=8)
        return ax
    
    else:
        plt.plot([scale_bar_x, scale_bar_x + scale_length], [scale_bar_y, scale_bar_y], color='hotpink', linewidth=2)
        plt.text(scale_bar_x + scale_length/2 - 0.046, scale_bar_y + 0.02, r'$\lambda / D$', color='hotpink', fontsize=8)


def get_residual_bounds(arr: Array):
    extent = np.nanmax(np.abs(arr))
    bound_dict = {
        "vmin": -extent,
        "vmax": extent,
        "cmap": seismic
    }
    return bound_dict


def get_residuals(arr1: Array, arr2: Array, return_bounds: bool = False):
    arr1 = np.array(arr1)
    arr2 = np.array(arr2)
    residuals = (arr1 - arr2)

    if return_bounds:
        bound_dict = get_residual_bounds(residuals)
        return residuals, bound_dict
    
    return residuals


def niriss_parang(hdr):
    """
    Stolen from AMICAL!
    """
    if hdr is None:
        warnings.warn(
            "No SCI header for NIRISS. No PA correction will be applied.",
            RuntimeWarning,
            stacklevel=2,
        )
        return 0.0
    v3i_yang = hdr["V3I_YANG"]  # Angle from V3 axis to ideal y axis (deg)
    roll_ref_pa = hdr["ROLL_REF"]  # Offset between V3 and N in local aperture coord

    return roll_ref_pa - v3i_yang


def get_extents(data, pixel_scale: float = None):
    x = data.shape[0]
    if pixel_scale is not None:
        x *= pixel_scale
    return {"extent": (-x/2, x/2, -x/2, x/2)}


def add_noise_to_ramp(clean_ramp):
    """
    Add poisson noise to the ramp.
    """
    ramp = []
    for i in range(len(clean_ramp)):
        seed = onp.random.randint(0, 100000)  # random seed for PRNGKey
        # if first group, just add poisson noise
        if i == 0:
            diff = jr.poisson(jr.PRNGKey(seed), clean_ramp[i])
            ramp.append(diff)
        # if not first group, add poisson noise on the new photons
        else:
            diff = jr.poisson(jr.PRNGKey(seed), clean_ramp[i] - clean_ramp[i - 1])
            ramp.append(ramp[-1] + diff)

    return np.array(ramp)
