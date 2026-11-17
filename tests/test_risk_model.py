from dataclasses import replace

import numpy as np

from risk_model import forecast_risk


def test_default_pipeline_and_future_data_isolation(make_panel):
    panel, oracle = make_panel(observations=160, stocks=12, seed=23)
    origin = panel.dates[-2]
    forecast = forecast_risk(panel, origin)
    assert forecast.shape == (12, 12)
    assert np.isfinite(forecast).all()
    np.testing.assert_allclose(forecast, forecast.T)
    assert np.linalg.eigvalsh(forecast).min() >= -1e-12
    relative_error = np.linalg.norm(forecast - oracle[-1]) / np.linalg.norm(oracle[-1])
    assert relative_error < 2
    altered_fields = {}
    for field in ("returns", "style_values", "market_caps", "industry_membership"):
        values = getattr(panel, field).copy()
        values[-1] = -999
        altered_fields[field] = values
    altered = replace(panel, **altered_fields)
    np.testing.assert_array_equal(forecast_risk(altered, origin), forecast)
