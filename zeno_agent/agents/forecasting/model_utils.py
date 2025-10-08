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
    data_size = len(df)
    has_monthly = df["ds"].dt.to_period("M").nunique() >= 12

    if PROPHET_AVAILABLE and has_monthly and data_size >= 12:
        try:
            m = Prophet(interval_width=0.8)
            m.fit(df[["ds", "y"]])
            future = m.make_future_dataframe(periods=periods, freq="MS")
            forecast_df = m.predict(future)
            point_forecast = forecast_df["yhat"].tail(periods).tolist()
            intervals = list(zip(
                forecast_df["yhat_lower"].tail(periods).tolist(),
                forecast_df["yhat_upper"].tail(periods).tolist()
            ))
            return point_forecast, intervals, "Prophet"
        except Exception:
            pass

    if ARIMA_AVAILABLE and data_size >= 8:
        try:
            model = ARIMA(df["y"], order=(1, 1, 1))
            fitted = model.fit()
            forecast_res = fitted.get_forecast(steps=periods)
            return (
                forecast_res.predicted_mean.tolist(),
                list(zip(forecast_res.conf_int().iloc[:, 0], forecast_res.conf_int().iloc[:, 1])),
                "ARIMA(1,1,1)"
            )
        except Exception:
            pass

    last_val = df["y"].iloc[-1]
    trend = (df["y"].iloc[-1] - df["y"].iloc[0]) / max(len(df) - 1, 1)
    return [last_val + trend * (i + 1) for i in range(periods)], [(None, None)] * periods, "Linear Trend"
