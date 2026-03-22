import numpy as np

from diffusion_mol.scaling import TemperatureScaler, inv_log10_d, log10_d


def test_log10_roundtrip():
    d = np.array([1e-6, 1e-5])
    np.testing.assert_allclose(inv_log10_d(log10_d(d)), d)


def test_temperature_scaler_fit_transform():
    s = TemperatureScaler()
    train_t = np.array([273.0, 298.0, 373.0])
    s.fit(train_t)
    x = s.transform(np.array([300.0, 400.0]))
    assert x.shape == (2,)
