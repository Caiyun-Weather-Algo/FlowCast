import torch
import torch.fft
import typing as t


class ZonalEnergySpectrum:
    def __init__(self, lon: torch.Tensor, lat: torch.Tensor):
        """
        lon: shape (W,), longitude in degrees (e.g., 0 to 359)
        lat: shape (H,), latitude in degrees (-90 to 90)
        """
        self.lon = lon
        self.lat = lat
        self.earth_radius = 6371000.0  # meters
        # circumference
        circum_at_equator = 2 * torch.pi * self.earth_radius
        self._circumference = torch.cos(self.lat * torch.pi / 180) * circum_at_equator
        # lon spacing in degree
        self.lon_spacing_deg = self.lon[1] - self.lon[0]
        if not torch.allclose((self.lon[1:] - self.lon[:-1]), self.lon_spacing_deg.expand_as(self.lon[:-1]), atol=1e-3):
            raise ValueError("Longitude must be uniformly spaced")
        self.lon_spacing_m = self._circumference * self.lon_spacing_deg / 360

    def compute(self, x: torch.Tensor) -> t.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, C, H, W) tensor
        Returns:
            spectra: (B, C, K)
            frequency: (K,)
            wavelength: (K,)
        """
        B, C, H, W = x.shape

        # FFT along longitude
        f_k = torch.fft.rfft(x, dim=-1, norm='forward')  # (B, C, H, K)

        K = f_k.shape[-1]
        one_and_many_twos = torch.ones(K, device=x.device)
        one_and_many_twos[1:] = 2

        # Energy spectrum per latitude
        power = (f_k.real ** 2 + f_k.imag ** 2) * one_and_many_twos  # (B, C, H, K)

        circumference = self._circumference[None, None, :, None].to(x.device)  # (1, 1, H, 1)
        power = power * circumference

        spectrum = power.mean(dim=2)  # (B, C, K) # latitude average

        base_frequency = torch.fft.rfftfreq(n=W, d=float(self.lon_spacing_deg)).to(x.device)
        frequency = base_frequency[None, :] / self.lon_spacing_m[:, None]
        wavelength = 1 / frequency
        return spectrum, frequency, wavelength


def compute_spectrum(x):
    """
    Compute the zonal energy spectrum of a tensor.
    """
    lon = torch.linspace(0, 359, 360).to(x.device)
    lat = torch.linspace(-90, 90, 181).to(x.device)
    spectrum_model = ZonalEnergySpectrum(lon, lat)
    spectrum, frequency, wavelength = spectrum_model.compute(x)
    return torch.log10(spectrum)


if __name__ == '__main__':
    torch.manual_seed(42)
    x = torch.randn(1, 2, 181, 360, requires_grad=True)
    lon = torch.linspace(0, 359, 360)
    lat = torch.linspace(-90, 90, 181)

    spectrum_model = ZonalEnergySpectrum(lon, lat)
    spectrum, frequency, wavelength = spectrum_model.compute(x)  # shape: (4, 20, 181)
    print(spectrum.shape)
    print(torch.log10(spectrum[0, :, 0:5]))
    loss = spectrum.mean()
    loss.backward()