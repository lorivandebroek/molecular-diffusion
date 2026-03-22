import numpy as np

from diffusion_mol.evaluate import regression_metrics
from diffusion_mol.plot import plot_benchmark_bars, plot_parity_log, plot_residuals_vs_temperature


def test_regression_metrics_keys():
    y = np.log10(np.array([1e-5, 2e-5, 3e-5]))
    p = y + 0.01
    m = regression_metrics(y, p)
    assert "mae_log10" in m and "rmse_D" in m


def test_plots_write(tmp_path):
    y = np.random.randn(20)
    p = y + 0.05 * np.random.randn(20)
    plot_parity_log(y, p, tmp_path / "parity.png")
    plot_residuals_vs_temperature(np.linspace(270, 380, 20), y, p, tmp_path / "res.png")
    plot_benchmark_bars({"a": 0.1, "b": 0.2}, tmp_path / "bar.png")
    assert (tmp_path / "parity.png").is_file()
    assert (tmp_path / "res.png").is_file()
    assert (tmp_path / "bar.png").is_file()
