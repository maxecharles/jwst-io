import dLux
from dLux.sources import Source
from dLux.psfs import PSF
from dLux import utils as dlu

from jax import Array, vmap, numpy as np, random as jr, scipy as jsp
from jax.scipy.signal import convolve
import equinox as eqx

from astropy import units as u

import numpy as onp
from numpy import random as rd

from amigo.misc import planck
from amigo.detector_layers import model_ramp
from amigo.stats import total_amplifier_noise

from matplotlib import colormaps

seismic = colormaps["seismic"]
seismic.set_bad("k", 0.5)

# plt.rcParams["font.size"] = 7

Optics = lambda: dLux.optical_systems.BaseOpticalSystem
Spectrum = lambda: dLux.spectra.BaseSpectrum


class BaseIoSource(Source):
    log_flux: Array | float
    position: Array

    def __init__(
        self: Source,
        position: Array | tuple = np.zeros(2),  # arcseconds
        log_flux: Array | float = 7.0,
        wavelengths: Array = None,
        weights: Array = None,
        spectrum: Spectrum() = None,
    ):
        self.position = np.array(position, dtype=float)  # arcseconds
        self.log_flux = np.array(log_flux, dtype=float)

        super().__init__(wavelengths=wavelengths, weights=weights, spectrum=spectrum)


class SimpleIoSource(BaseIoSource):
    log_distribution: Array

    def __init__(
        self: Source,
        position: Array | tuple = np.zeros(2),  # arcseconds
        log_flux: Array | float = 7.0,
        log_distribution: Array = None,
        wavelengths: Array = None,
        weights: Array = None,
        spectrum: Spectrum() = None,
    ):
        self.log_distribution = np.array(log_distribution, dtype=float)

        super().__init__(
            position=position,
            log_flux=log_flux,
            wavelengths=wavelengths,
            weights=weights,
            spectrum=spectrum,
        )

    @property
    def distribution(self: Source) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        distribution = np.power(10, self.log_distribution)
        return distribution / distribution.sum()

    def model(
        self: Source,
        optics: Optics = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
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

        distribution = np.power(10, self.log_distribution)
        distribution = distribution / distribution.sum()

        # Returning wf is a special case
        if return_wf:
            conv_fn = lambda psf: convolve(psf, distribution, mode="same")
            return wf.set("amplitude", vmap(conv_fn)(wf.psf) ** 0.5)

        # Return psf object
        conv_psf = convolve(wf.psf.sum(0), distribution, mode="same")
        if return_psf:
            return PSF(conv_psf, wf.pixel_scale.mean())

        # Return array psf
        return conv_psf


class ComplexIoSource(BaseIoSource):
    volc_frac: Array | float
    log_volcanoes: Array
    disk: Array

    def __init__(
        self: Source,
        position: Array | tuple = np.zeros(2),  # arcseconds
        log_flux: Array | float = 7.0,
        volc_frac: Array | float = 1e-2,
        log_volcanoes: Array = None,
        disk: Array = None,
        wavelengths: Array = None,
        weights: Array = None,
        spectrum: Spectrum() = None,
    ):
        if disk is None:
            self.disk = dLux.PSF(
                initialise_disk(0.065524085, 4, normalise=True), 0.065524085
            )
        else:
            self.disk = disk

        if log_volcanoes is None:
            self.log_volcanoes = dLux.PSF(
                initialise_disk(0.065524085, 4, normalise=True), 0.065524085
            ).data
        else:
            self.log_volcanoes = np.array(log_volcanoes)

        self.position = np.array(position, dtype=float)  # arcseconds
        self.log_flux = np.array(log_flux, dtype=float)
        self.volc_frac = np.array(volc_frac, dtype=float)

        super().__init__(
            position=position,
            log_flux=log_flux,
            wavelengths=wavelengths,
            weights=weights,
            spectrum=spectrum,
        )

    @property
    def volcanoes(self: Source) -> Array:
        """
        Returns the volcano distribution.
        """
        volcanoes = np.power(10, np.array(self.log_volcanoes))
        return volcanoes / volcanoes.sum()

    @property
    def distribution(self: Source) -> Array:
        """
        Assuming the disk and volcanoes array sum to 1, this property returns the
        source intensity distribution which also sums to 1.
        """
        return (1.0 - self.volc_frac) * self.disk.data + self.volc_frac * self.volcanoes

    def model(
        self: Source,
        optics: Optics = None,
        return_wf: bool = False,
        return_psf: bool = False,
    ) -> Array:
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

        volcanoes = np.power(10, np.array(self.log_volcanoes))
        volcanoes /= volcanoes.sum()
        distribution = (
            1.0 - self.volc_frac
        ) * self.disk.data + self.volc_frac * volcanoes

        # Returning wf is a special case
        if return_wf:
            conv_fn = lambda psf: convolve(psf, distribution, mode="same")
            return wf.set("amplitude", vmap(conv_fn)(wf.psf) ** 0.5)

        # Return psf object
        conv_psf = convolve(wf.psf.sum(0), distribution, mode="same")
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


class DefinitelyRealIo:
    def __init__(
        self,
        n_volcanoes=None,
        night_day_ratio=None,
        terminator_angle=None,  # degrees
        darkness=None,
        R=1.1519 / 2,
        R_volc=1e-2,
        volc_contrast=1.0,
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

        self.disk = dLux.PSF(
            initialise_disk(pixel_scale, oversample, normalise=True), pixel_scale
        )

        ill_mask = self.generate_illumination_mask(
            night_day_ratio, terminator_angle, self.disk.data.shape[0]
        )
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
                x_rot = x * onp.cos(onp.radians(rotation_angle)) - y * onp.sin(
                    onp.radians(rotation_angle)
                )
                y_rot = x * onp.sin(onp.radians(rotation_angle)) + y * onp.cos(
                    onp.radians(rotation_angle)
                )

                # Check if the point falls within the ellipse
                if x_rot**2 + y_rot**2 <= semi_major**2:
                    if onp.arctan2(y_rot, x_rot) > 0:
                        if (x_rot / semi_major) ** 2 + (y_rot / semi_minor) ** 2 >= 1:
                            array[i, j] = True
                    elif onp.arctan2(y_rot, x_rot) < 0:
                        array[i, j] = False

        return array

    def generate_volcanoes(self):
        disk = dLux.PSF(
            initialise_disk(self.pixel_scale, self.oversample), self.pixel_scale
        )

        # generating cartesian coordinates of volcanoes
        volcano_coords = np.array(
            [np.power(rd.uniform(size=(self.n_volcanoes,)), p) for p in [0.5, 1]]
        ).T
        volcano_coords *= np.array(
            self.n_volcanoes
            * [
                [self.R, 2 * np.pi],
            ]
        )
        volcano_coords = dlu.polar2cart(volcano_coords.T).T

        # volcano brightness
        volcano_brights = onp.abs(
            rd.normal(loc=1.0, scale=0.3, size=(self.n_volcanoes,))
        )
        volcano_brights = np.array(
            [np.ones_like(disk.data) * bright for bright in volcano_brights]
        )

        # generating coordinate grids for each volcano
        coords = dlu.pixel_coords(
            npixels=disk.data.shape[0],
            diameter=disk.data.shape[0] * self.pixel_scale / self.oversample,
        )
        coords = [
            dlu.translate_coords(coords, volcano_coords[i])
            for i in range(self.n_volcanoes)
        ]

        # creating an array for each volcano and summing to have an array with all volcanoes
        volcanoes = np.multiply(
            np.array(
                [
                    dlu.soft_circle(coord, self.R_volc, clip_dist=1e-2)
                    for coord in coords
                ]
            ),
            volcano_brights,
        ).sum(0)

        return volcanoes / volcanoes.sum()

    @property
    def data(self):
        return (
            np.where(self.ill_mask, self.disk.data * self.darkness, self.disk.data)
            + self.volc_contrast * self.volcanoes
        )


def io_model_fn(model, exposure, with_BFE=True, to_BFE=False, zero_idx=-1, noise=True):
    # Get exposure key
    key = exposure.key

    # Updating source weights
    wavels, filt_weights = model.filters[exposure.filter]
    if exposure.star == "IO":
        weights = filt_weights  # TODO spectrum
    elif exposure.star == "PSFCAL.2022A-HD2236-K6":
        weights = filt_weights * planck(wavels, model.Teffs[exposure.star])
    # weights *= 10 ** (model.log_fluxes[key]) / weights.sum()

    source = model.source.set(
        ["position", "log_flux", "wavelengths", "weights"],
        [model.positions[key], model.log_fluxes[key], wavels, weights],
    )

    # Apply correct aberrations
    aberrations = model.aberrations[key]
    if zero_idx != -1:
        aberrations = aberrations.at[zero_idx, 0].set(0.0)  # Pin piston to zero

    optics = model.optics.set(
        ["coefficients", "pupil.opd"],
        [aberrations, exposure.opd],
    )

    PSF = source.model(optics, return_psf=True)

    # Apply the detector model and turn it into a ramp
    psf = model.detector.model(PSF)
    ramp = model_ramp(psf, exposure.ngroups)
    if to_BFE:
        return ramp

    # Now apply the CNN BFE and downsample
    if with_BFE:
        ramp = eqx.filter_vmap(model.BFE.apply_array)(ramp)
    else:
        dsample_fn = lambda x: dlu.downsample(x, 4, mean=False)
        ramp = vmap(dsample_fn)(ramp)
    ramp = vmap(dlu.resize, (0, None))(ramp, 80)

    # Apply bias and one of F correction
    if noise:
        # ramp += total_read_noise(model.biases[key], model.one_on_fs[key])
        ramp += total_amplifier_noise(model.one_on_fs[key])

    # return ramp
    return np.diff(ramp, axis=0)


def sim_io_model_fn(model, ngroups, to_BFE=False, noise=True):
    source = model.source
    optics = model.optics

    PSF = source.model(optics, return_psf=True)

    # Apply the detector model and turn it into a ramp
    psf = model.detector.model(PSF)
    ramp = model_ramp(psf, ngroups)

    if to_BFE:
        return ramp

    # Now apply the CNN BFE and downsample
    ramp = eqx.filter_vmap(model.BFE.apply_array)(ramp)
    ramp = vmap(dlu.resize, (0, None))(ramp, 80)

    # Apply bias and one of F correction
    if noise:
        # ramp += total_read_noise(model.biases[key], model.one_on_fs[key])
        ramp += total_amplifier_noise(model.one_on_fs)

    # return ramp
    return np.diff(ramp, axis=0)


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


def get_filter_spectrum(filt: str, file_path: str, nwavels: int = 9):
    if filt not in ["F380M", "F430M", "F480M", "F277W"]:
        raise ValueError("Supported filters are F380M, F430M, F480M, F277W.")

    # filter_path = os.path.join()
    wl_array, throughput_array = np.array(
        onp.loadtxt(file_path + f"JWST_NIRISS.{filt}.dat", unpack=True)
    )

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


def diff_lim_to_cov(model):
    lambda_on_D = dlu.rad2arcsec(
        model.source.wavelengths.mean() / model.optics.diameter
    )
    sigma = 1.025 / (2 * np.sqrt(2 * np.log(2))) * lambda_on_D
    return [[sigma**2, 0], [0, sigma**2]]


def get_pscale(model):
    return model.optics.psf_pixel_scale / model.optics.oversample


def blur_distribution(model, extent=0.15):
    cov = diff_lim_to_cov(model)

    x = np.arange(-extent, extent, get_pscale(model))
    X, Y = np.meshgrid(x, x)
    pos = np.dstack((X, Y))

    kernel = jsp.stats.multivariate_normal.pdf(jr.PRNGKey(0), pos, np.array(cov))

    distribution = jsp.signal.convolve2d(model.distribution, kernel, mode="same")
    return distribution / distribution.sum()
