"""Top-level risk forecast using the module defaults."""

from dataclasses import replace

import numpy as np

from ._validation import _dates
from .dynamic_risk import estimate_dynamic_variance
from .factor_model import fit_factor_model
from .integration import compute_asset_covariance, rescale_covariance
from .stable_risk import estimate_factor_covariance


def forecast_risk(panel, as_of):
    """Return next-observation stock covariance using history through as_of."""
    dates = _dates(panel.dates)
    cutoff = np.datetime64(as_of, "ns")
    if np.isnat(cutoff):
        raise ValueError("as_of must be a valid date")
    count = int(np.searchsorted(dates, cutoff, side="right"))
    histories = {}
    for field in ("returns", "style_values", "industry_membership", "market_caps"):
        values = np.asarray(getattr(panel, field))
        if values.ndim == 0 or len(values) != len(dates):
            raise ValueError(f"{field} must share the panel date axis")
        histories[field] = values[:count]
    truncated = replace(panel, dates=dates[:count], **histories)
    factors = fit_factor_model(truncated)
    stable = estimate_factor_covariance(factors.factor_returns)
    factor_dynamic = estimate_dynamic_variance(factors.factor_returns)
    specific_dynamic = estimate_dynamic_variance(factors.residuals)
    factor_covariance = rescale_covariance(
        stable, np.sqrt(factor_dynamic.forecast_variance)
    )
    return compute_asset_covariance(
        factors.exposures[-1], factor_covariance, specific_dynamic.forecast_variance
    )
