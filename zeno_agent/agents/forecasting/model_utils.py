"""
Forecasting model utilities for time series prediction.

Implements a hierarchical model selection strategy:
1. Prophet (Facebook's forecasting tool) - for sufficient monthly data
2. ARIMA - for shorter time series
3. Linear Trend - fallback for minimal data

All models return point forecasts, confidence intervals, and model identifiers.
"""

import numpy as np
from typing import Tuple, List

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False


def run_model(df, periods: int, metric_type: str) -> Tuple[List[float], List[Tuple[float, float]], str]:
    """
    Run appropriate forecasting model based on data characteristics.
    
    Args:
        df: DataFrame with 'ds' (datetime) and 'y' (target variable) columns
        periods: Number of periods to forecast ahead
        metric_type: Description of what's being forecasted (for logging/debugging)
    
    Returns:
        Tuple of (point_forecasts, confidence_intervals, model_name)
        - point_forecasts: List of predicted values
        - confidence_intervals: List of (lower, upper) tuples
        - model_name: String identifier of the model used
    """
    data_size = len(df)
    has_monthly = df["ds"].dt.to_period("M").nunique() >= 12

    # Strategy 1: Prophet for robust monthly data (preferred)
    if PROPHET_AVAILABLE and has_monthly and data_size >= 12:
        try:
            model = Prophet(
                interval_width=0.8,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05  # Conservative changepoint detection
            )
            model.fit(df[["ds", "y"]])
            future = model.make_future_dataframe(periods=periods, freq="MS")
            forecast_df = model.predict(future)
            
            point_forecast = forecast_df["yhat"].tail(periods).tolist()
            intervals = list(zip(
                forecast_df["yhat_lower"].tail(periods).tolist(),
                forecast_df["yhat_upper"].tail(periods).tolist()
            ))
            return point_forecast, intervals, "Prophet (Bayesian)"
        except Exception:
            # Prophet failed, fall through to next strategy
            pass

    # Strategy 2: ARIMA for moderate-sized time series
    if ARIMA_AVAILABLE and data_size >= 8:
        try:
            # ARIMA(1,1,1) is a reasonable default for many economic series
            # p=1: one autoregressive term
            # d=1: first-order differencing (handles non-stationarity)
            # q=1: one moving average term
            model = ARIMA(df["y"], order=(1, 1, 1))
            fitted = model.fit()
            forecast_res = fitted.get_forecast(steps=periods)
            
            return (
                forecast_res.predicted_mean.tolist(),
                list(zip(
                    forecast_res.conf_int().iloc[:, 0].tolist(),
                    forecast_res.conf_int().iloc[:, 1].tolist()
                )),
                "ARIMA(1,1,1)"
            )
        except Exception:
            # ARIMA failed (common with very short series), fall through
            pass

    # Strategy 3: Simple linear trend (fallback for minimal data)
    # This is a last resort and should be interpreted cautiously
    last_val = df["y"].iloc[-1]
    first_val = df["y"].iloc[0]
    trend = (last_val - first_val) / max(len(df) - 1, 1)
    
    forecasts = [last_val + trend * (i + 1) for i in range(periods)]
    
    # No confidence intervals for linear trend (too unreliable)
    intervals = [(None, None)] * periods
    
    return forecasts, intervals, "Linear Trend (fallback)"


def validate_forecast_quality(df, forecast: List[float], model_name: str) -> dict:
    """
    Assess forecast quality based on data characteristics.
    
    Returns a dictionary with quality metrics for logging or user feedback.
    """
    data_size = len(df)
    historical_mean = df["y"].mean()
    historical_std = df["y"].std()
    
    # Coefficient of variation (volatility indicator)
    cv = (historical_std / historical_mean * 100) if historical_mean > 0 else 0
    
    # Assess quality
    if model_name.startswith("Prophet") and data_size >= 24:
        quality = "High"
        reliability = "Forecast based on robust seasonal decomposition with >2 years data"
    elif model_name.startswith("ARIMA") and data_size >= 12:
        quality = "Medium-High"
        reliability = "Forecast based on time series modeling with adequate historical data"
    elif data_size >= 8:
        quality = "Medium"
        reliability = "Forecast based on limited historical data; interpret with caution"
    else:
        quality = "Low"
        reliability = "Forecast based on minimal data; use as rough estimate only"
    
    return {
        "quality": quality,
        "reliability": reliability,
        "data_points": data_size,
        "volatility_pct": round(cv, 1),
        "model": model_name
    }