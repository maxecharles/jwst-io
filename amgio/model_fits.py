from amigo.model_fits import ModelFit
from jax import Array


class ResolvedFit(ModelFit):
    def get_distribution(self, model, exposure) -> Array:
        """
        Returns the normalised intensity distribution of the source.
        """
        distribution = (
            10 ** model.log_distribution[self.get_key(exposure, "log_distribution")]
        )
        return distribution / distribution.sum()

    def get_key(self, exposure, param):
        match param:
            case "log_distribution":
                return exposure.key

        return super().get_key(exposure, param)

    def map_param(self, exposure, param):
        match param:
            case "log_distribution":
                return f"{param}.{exposure.get_key(param)}"

        return super().map_param(exposure, param)

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        image = psf.convolve(self.get_distribution(model, exposure))
        image = self.model_detector(image, model, exposure)
        ramp = self.model_ramp(image, model, exposure)
        return self.model_read(ramp, model, exposure)
    

class IoResolvedFit(ResolvedFit):

    def get_spectra(self, model, exposure):
        wavels, filt_weights = model.filters[exposure.filter]
        source_weights = model.source_spectrum.weights
        weights = filt_weights * (source_weights / source_weights.sum())
        return wavels, weights / weights.sum()

    def get_key(self, exposure, param):
        match param:
            case "spectral_coeffs":
                return exposure.program

        return super().get_key(exposure, param)

    def map_param(self, exposure, param):
        match param:
            case "spectral_coeffs":
                return f"{param}.{exposure.get_key(param)}"

        return super().map_param(exposure, param)