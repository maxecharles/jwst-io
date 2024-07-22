from amigo.core_models import BaseModeller
from amigo.optical_models import AMIOptics
from amigo.detector_models import LinearDetectorModel
from amigo.ramp_models import SimpleRamp
from amigo.read_models import ReadModel
from amigo.files import get_filters
from jax import numpy as np
import dLux as dl


class ResolvedAmigoModel(BaseModeller):
    filters: dict
    dispersion: dict
    contrast: float
    optics: AMIOptics
    visibilities: None
    detector: None
    ramp: None
    read: None

    def __init__(
        self,
        files,
        params,
        optics=None,
        ramp=None,
        detector=None,
        read=None,
        visibilities=None,
        dispersion_mag=0.0,  # arcseconds
        contrast=-2,
    ):
        if optics is None:
            optics = AMIOptics()
        if detector is None:
            detector = LinearDetectorModel()
        if ramp is None:
            ramp = SimpleRamp()
        if read is None:
            read = ReadModel()

        self.filters = get_filters(files)

        # Dispersion hacking - randomly perturb the position of each wavelength
        if dispersion_mag > 0.0:
            self.dispersion = {}

            # # This one is free-floating value per wavelength
            # for filt, (wavels, weights) in self.filters.items():
            #     rand_positions = jr.normal(jr.PRNGKey(0), (len(wavels), 2))
            #     self.dispersion[filt] = dispersion_mag * rand_positions

            # This one is parameterised by (x, y) - the point at which the longest
            # wavelength reaches
            for filt in self.filters.keys():
                self.dispersion[filt] = np.array([dispersion_mag, dispersion_mag])

        else:
            self.dispersion = None
        self.contrast = np.asarray(contrast, float)

        self.optics = optics
        self.detector = detector
        self.ramp = ramp
        self.read = read
        self.visibilities = visibilities
        self.params = params

    def distribution(self, exposure):
        distribution = 10 ** self.log_distribution[exposure.key]
        return distribution / distribution.sum()

    def model(self, exposure, **kwargs):
        return exposure.fit(self, exposure, **kwargs)

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        if hasattr(self.optics, key):
            return getattr(self.optics, key)
        if hasattr(self.detector, key):
            return getattr(self.detector, key)
        if hasattr(self.ramp, key):
            return getattr(self.ramp, key)
        if hasattr(self.read, key):
            return getattr(self.read, key)
        if hasattr(self.visibilities, key):
            return getattr(self.visibilities, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class IoAmigoModel(ResolvedAmigoModel):
    source_spectrum: dl.PolySpectrum

    def __init__(
        self,
        files,
        params,
        optics=None,
        ramp=None,
        detector=None,
        read=None,
        source_coeffs=None,
        visibilities=None,
        dispersion_mag=0.0,  # arcseconds
        contrast=-2,
    ):
        super().__init__(
            files,
            params,
            optics=optics,
            ramp=ramp,
            detector=detector,
            read=read,
            visibilities=visibilities,
            dispersion_mag=dispersion_mag,
            contrast=contrast,
        )

        wavels = self.filters["F430M"][0]
        if source_coeffs is None:
            source_coeffs = np.array([1.0, 0.0])

        self.source_spectrum = dl.PolySpectrum(wavels, source_coeffs)
