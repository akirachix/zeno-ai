import os
import time
import json
import threading
import re
import pickle
from typing import Dict, Any, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import datetime
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_crop_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    query_rag_embeddings_semantic
)

from zeno_agent.embedding_utils import encode_query_to_vector
_MODEL_CACHE: Dict[str, Tuple[float, Any]] = {}
_RESULT_CACHE: Dict[str, Tuple[float, Dict[str, Any]]] = {}
_CACHE_TTL = 3600 
_cache_lock = threading.Lock()

redis_client = None
if REDIS_AVAILABLE:
    REDIS_URL = os.getenv("REDIS_URL")
    if REDIS_URL:
        try:
            redis_client = redis.from_url(REDIS_URL)
        except Exception as e:
            print(f"Redis connection failed: {e}")

def _cache_model_key(country, commodity, metric, model_type) -> str:
    return f"model::{country}::{commodity}::{metric}::{model_type}"

def _cache_result_key(country, commodity, metric, timeframe, model_type, rag_key="") -> str:
    return f"result::{country}::{commodity}::{metric}::{timeframe}::{model_type}::{rag_key}"

def cache_get_model(key: str) -> Optional[Any]:
    if redis_client:
        data = redis_client.get(f"forecast:model:{key}")
        return pickle.loads(data) if data else None
    else:
        entry = _MODEL_CACHE.get(key)
        if entry and (time.time() - entry[0] < _CACHE_TTL):
            return entry[1]
        return None

def cache_set_model(key: str, model_obj: Any) -> None:
    if redis_client:
        try:
            redis_client.setex(f"forecast:model:{key}", _CACHE_TTL, pickle.dumps(model_obj))
        except Exception as e:
            print(f"Redis model cache set failed: {e}")
    else:
        with _cache_lock:
            _MODEL_CACHE[key] = (time.time(), model_obj)

def cache_get_result(key: str) -> Optional[Dict[str, Any]]:
    if redis_client:
        data = redis_client.get(f"forecast:result:{key}")
        return pickle.loads(data) if data else None
    else:
        entry = _RESULT_CACHE.get(key)
        if entry and (time.time() - entry[0] < _CACHE_TTL):
            return entry[1]
        return None

def cache_set_result(key: str, result: Dict[str, Any]) -> None:
    if redis_client:
        try:
            redis_client.setex(f"forecast:result:{key}", _CACHE_TTL, pickle.dumps(result))
        except Exception as e:
            print(f"Redis result cache set failed: {e}")
    else:
        with _cache_lock:
            _RESULT_CACHE[key] = (time.time(), result)


class ForecastingAgent:
    def __init__(self):
        self.commodity_mapping = {
            "maize": "maize",
            "coffee": "coffee",
            "tea": "tea"
        }
        self.supported_countries = ["kenya", "ethiopia", "rwanda"]
        self.supported_models = ["ARIMA", "Prophet", "XGBoost", "Ensemble"]

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        if df['y'].dtype == 'object':
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df = df.dropna(subset=['y'])
        df['y'] = df['y'].interpolate(method='linear').ffill().bfill()
        df = df[np.abs(df['y'] - df['y'].mean()) <= (3 * df['y'].std())]
        df['lag_1'] = df['y'].shift(1)
        df['lag_12'] = df['y'].shift(12)
        df = df.dropna()
        return df.sort_values('ds').reset_index(drop=True)

    def run_prophet(self, df: pd.DataFrame, periods: int):
        m = Prophet(
            mcmc_samples=0,
            uncertainty_samples=0,
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='multiplicative'
        )
        m.fit(df[['ds', 'y']])
        future = m.make_future_dataframe(periods=periods, freq='MS')
        forecast_df = m.predict(future)
        forecast_series = forecast_df['yhat'].tail(periods).tolist()
        confidence_intervals = list(zip(forecast_df['yhat_lower'].tail(periods), forecast_df['yhat_upper'].tail(periods)))
        return forecast_series, confidence_intervals

    def run_arima(self, df: pd.DataFrame, periods: int):
        model = ARIMA(df['y'], order=(1,1,1))
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=periods)
        forecast_series = forecast_result.predicted_mean.tolist()
        conf_int = forecast_result.conf_int(alpha=0.05)
        confidence_intervals = list(zip(conf_int.iloc[:, 0], conf_int.iloc[:, 1]))
        return forecast_series, confidence_intervals

    def run_xgboost(self, df: pd.DataFrame, periods: int):
        df = df.copy()
        df['lag_1'] = df['y'].shift(1)
        df['lag_12'] = df['y'].shift(12)
        df = df.dropna()
        X = df[['lag_1', 'lag_12']]
        y = df['y']
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        future_X = pd.DataFrame({
            'lag_1': [df['y'].iloc[-1]] * periods,
            'lag_12': [df['y'].iloc[-12]] * periods if len(df) >= 12 else [df['y'].iloc[-1]] * periods
        })
        forecast_series = model.predict(future_X).tolist()
        confidence_intervals = [("low", "high")] * periods
        return forecast_series, confidence_intervals

    def get_or_train_model(self, key: str, df: pd.DataFrame, model_type: str):
        cached = cache_get_model(key)
        if cached is not None:
            return cached

        if model_type == "Prophet":
            m = Prophet(
                mcmc_samples=0,
                uncertainty_samples=0,
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative'
            )
            m.fit(df[['ds', 'y']])
            cache_set_model(key, m)
            return m
        elif model_type == "ARIMA":
            model = ARIMA(df['y'], order=(1,1,1))
            fitted = model.fit()
            cache_set_model(key, fitted)
            return fitted
        elif model_type == "XGBoost":
            X = df[['lag_1', 'lag_12']].dropna()
            y = df.loc[X.index, 'y']
            m = XGBRegressor(n_estimators=100, learning_rate=0.1)
            m.fit(X, y)
            cache_set_model(key, m)
            return m
        else:
            return None

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        run_start = time.time()
        try:
            if isinstance(params, dict):
                query_text = params.get("query", "")
            else:
                query_text = str(params) if params is not None else ""
            if not isinstance(query_text, str):
                query_text = str(query_text)

            commodity = params.get("commodity")
            metric = params.get("metric") 
            country = params.get("country")
            timeframe = params.get("timeframe", "next 3 months")
            model_type = params.get("model_type")

            query_lower = query_text.lower()
            if commodity is None:
                if "maize" in query_lower:
                    commodity = "maize"
                elif "coffee" in query_lower:
                    commodity = "coffee"
                elif "tea" in query_lower:
                    commodity = "tea"
                else:
                    commodity = "maize"

            if metric is None:
                if "export" in query_lower and ("volume" in query_lower or "volumes" in query_lower):
                    metric = "export_volume"
                elif "price" in query_lower:
                    metric = "price"
                elif "revenue" in query_lower:
                    metric = "revenue"
                else:
                    metric = "price"

            if country is None:
                if "kenya" in query_lower:
                    country = "kenya"
                elif "ethiopia" in query_lower:
                    country = "ethiopia"
                elif "rwanda" in query_lower:
                    country = "rwanda"
                else:
                    country = "kenya"

            timeframe_lower = timeframe.lower().strip()
            if "next year" in timeframe_lower:
                months = 12
            elif "next quarter" in timeframe_lower:
                months = 3
            elif "next month" in timeframe_lower:
                months = 1
            else:
                year_match = re.search(r"(\d+)\s*(?:year|yr)s?", timeframe_lower)
                if year_match:
                    months = int(year_match.group(1)) * 12
                else:
                    month_match = re.search(r"(\d+)\s*(?:month|mo)s?", timeframe_lower)
                    months = int(month_match.group(1)) if month_match else 3

            try:
                country_id = get_country_id_by_name(country)
                product_id = get_crop_id_by_name(self.commodity_mapping.get(commodity, commodity))
                indicator_id = get_indicator_id_by_metric(metric)
            except Exception as e:
                return {
                    "error": f"Failed to resolve IDs: {str(e)}",
                    "debug": {"country": country, "commodity": commodity, "metric": metric},
                    "timings": {"total_seconds": time.time() - run_start}
                }

            df = get_trade_data_from_db(
                country_id=country_id,
                product_id=product_id,
                indicator_id=indicator_id,
                start_date="2015-01-01"
            )
            if df is None or df.empty:
                return {
                    "error": f"No historical data found for {commodity} {metric} in {country}.",
                    "debug": {"country_id": country_id, "product_id": product_id, "indicator_id": indicator_id},
                    "timings": {"total_seconds": time.time() - run_start}
                }

            df = df.copy()
            df['ds'] = pd.to_datetime(df['date'])
            value_col = 'price' if metric == "price" else 'quantity'
            df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
            df = df.dropna(subset=[value_col])
            df[value_col] = df[value_col].astype('float64')
            df = df[['ds', value_col]].rename(columns={value_col: 'y'}).sort_values('ds')
            if len(df) < 8:
                return {
                    "error": f"Insufficient data points ({len(df)}) for reliable forecasting.",
                    "timings": {"total_seconds": time.time() - run_start}
                }
            df_proc = self.preprocess_data(df)

            model_choice = model_type or ("ARIMA" if months <= 6 else "Prophet")

            rag_context = []
            try:
                query_embedding = encode_query_to_vector(f"{commodity} {metric} {country} forecast")
                rag_context = query_rag_embeddings_semantic(query_embedding)
            except Exception as e:
                print(f"RAG failed: {e}")

            rag_key = str(len(rag_context))
            result_key = _cache_result_key(country, commodity, metric, months, model_choice, rag_key)
            cached_result = cache_get_result(result_key)
            if cached_result:
                cached_result["cached"] = True
                cached_result["timings"] = {"total_seconds": time.time() - run_start}
                return cached_result

            model_cache_key = _cache_model_key(country, commodity, metric, model_choice)
            trained_model = self.get_or_train_model(model_cache_key, df_proc, model_choice)

            if model_choice == "Prophet":
                if trained_model is not None:
                    future = trained_model.make_future_dataframe(periods=months, freq='MS')
                    forecast_df = trained_model.predict(future)
                    forecast_series = forecast_df['yhat'].tail(months).tolist()
                else:
                    forecast_series, _ = self.run_prophet(df_proc, months)
            elif model_choice == "ARIMA":
                if trained_model is not None:
                    forecast_result = trained_model.get_forecast(steps=months)
                    forecast_series = forecast_result.predicted_mean.tolist()
                else:
                    forecast_series, _ = self.run_arima(df_proc, months)
            elif model_choice == "XGBoost":
                if trained_model is not None:
                    last_y = df_proc['y'].iloc[-1]
                    last_y12 = df_proc['y'].iloc[-12] if len(df_proc) >= 12 else last_y
                    future_X = pd.DataFrame({'lag_1': [last_y] * months, 'lag_12': [last_y12] * months})
                    forecast_series = trained_model.predict(future_X).tolist()
                else:
                    forecast_series, _ = self.run_xgboost(df_proc, months)
            else:
                forecasts = []
                for mtype in ["ARIMA", "Prophet", "XGBoost"]:
                    key = _cache_model_key(country, commodity, metric, mtype)
                    trained = cache_get_model(key)
                    try:
                        if mtype == "Prophet" and trained:
                            f = trained.make_future_dataframe(periods=months, freq='MS')
                            f = trained.predict(f)['yhat'].tail(months).tolist()
                        elif mtype == "ARIMA" and trained:
                            f = trained.get_forecast(steps=months).predicted_mean.tolist()
                        elif mtype == "XGBoost" and trained:
                            last_y = df_proc['y'].iloc[-1]
                            last_y12 = df_proc['y'].iloc[-12] if len(df_proc) >= 12 else last_y
                            future_X = pd.DataFrame({'lag_1': [last_y] * months, 'lag_12': [last_y12] * months})
                            f = trained.predict(future_X).tolist()
                        else:
                            f = [df_proc['y'].mean()] * months
                    except Exception:
                        f = [df_proc['y'].mean()] * months
                    forecasts.append(f)
                forecast_series = list(np.mean(np.array(forecasts), axis=0))

            variety_note = ""
            for doc in rag_context:
                txt = (doc.get("content") or "").lower()
                if ("sl28" in txt or "sl34" in txt) and ("disease" in txt or "cbd" in txt or "susceptible" in txt):
                    forecast_series = [x * 0.85 for x in forecast_series]  
                    variety_note = " Note: SL28/SL34 varieties (70% of production) are highly CBD-susceptible, which may impact yields."
                elif ("ruiru" in txt or "batian" in txt) and ("resistant" in txt or "resilient" in txt):
                    forecast_series = [x * 1.05 for x in forecast_series]  
                    variety_note = " Note: Disease-resistant Ruiru 11/Batian adoption may stabilize exports."
                elif "fertilizer" in txt and ("price" in txt or "cost" in txt):
                    if "71%" in txt or "surge" in txt:
                        forecast_series = [x * 0.9 for x in forecast_series]
                        variety_note = " Note: Record-high fertilizer prices (up 71%) are reducing production capacity."

            avg_forecast = float(np.mean(forecast_series))
            unit = {"price": "/kg", "export_volume": " tons", "revenue": " USD"}.get(metric, "")
            forecast_value = f"${avg_forecast:.2f}{unit}" if metric == "price" else f"{avg_forecast:,.0f}{unit}"

            result = {
                "forecast_value": forecast_value,
                "confidence": "Medium",
                "reasoning": f"Model: {model_choice}. Based on historical {commodity} {metric} data from {country}.{variety_note}",
                "model_used": model_choice,
                "forecast_series": forecast_series,
                "rag_context": rag_context,
                "metrics": {},
                "data_points": len(df_proc)
            }

            cache_set_result(result_key, result)
            result["timings"] = {"total_seconds": time.time() - run_start}
            return result

    def parse_timeframe(self, timeframe: str) -> int:
        import re
        match = re.match(r"next (\d+) (year|years|month|months)", timeframe.lower())
        if not match:
            return 3
        num, unit = int(match.group(1)), match.group(2)
        return num if "month" in unit else num * 12

    def forecast_dual_metrics(self, df, periods: int):
        unit_df = df[["ds", "unit_price"]].rename(columns={"unit_price": "y"})
        unit_forecast, unit_ints, unit_model = run_model(unit_df, periods, "unit_price")

        rev_df = df[["ds", "price"]].rename(columns={"price": "y"})
        rev_forecast, rev_ints, rev_model = run_model(rev_df, periods, "revenue")

        vol_df = df[["ds", "quantity_kg"]].rename(columns={"quantity_kg": "y"})
        vol_forecast, _, _ = run_model(vol_df, periods, "volume")

        return {
            "unit_price": {"forecast": np.mean(unit_forecast), "intervals": unit_ints, "model": unit_model},
            "total_revenue": {"forecast": np.mean(rev_forecast), "intervals": rev_ints, "model": rev_model},
            "volume_kg": np.mean(vol_forecast),
        }

    def run(self, inputs):
        query = inputs.get("query", "")
        if not query:
            return {"error": "No query provided."}

        q = query.lower()
        commodity = next((k for k in ALIAS_MAP if k in q), None)
        country = next((c for c in SUPPORTED_COUNTRIES if c in q), None)
        timeframe = "next 3 months" if "month" in q else "next 1 year"

        if not commodity or not country:
            return {"error": "Could not identify commodity or country."}

        commodity = ALIAS_MAP[commodity]
        country_id = get_country_id_by_name(country.title())
        product_id = get_product_id_by_name(commodity)
        df, currency, vol_unit, symbol = prepare_dual_data(country_id, product_id)
        rag_context = get_enhanced_rag_context(commodity, country, "price")

        periods = self.parse_timeframe(timeframe)
        dual_forecast = self.forecast_dual_metrics(df, periods)

        display_text = (
            f"Unit Price: {dual_forecast['unit_price']['forecast']:.2f} {currency}/kg | "
            f"Revenue: {dual_forecast['total_revenue']['forecast']:.0f} {currency} | "
            f"Volume: {dual_forecast['volume_kg']:.0f} {vol_unit}"
        )


        prompt = f"""
        Interpret this forecast professionally for economists.
        No markdown, no bullets, just structured paragraphs.

        Commodity: {commodity}
        Country: {country}
        Unit Price: {dual_forecast['unit_price']['forecast']:.2f} {currency}/kg
        Total Revenue: {dual_forecast['total_revenue']['forecast']:.0f} {currency}
        Volume: {dual_forecast['volume_kg']:.0f} {vol_unit}
        Context: {rag_context}
        {"Additional document context: " + file_context if file_context else ""}
        """

        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            interpretation = re.sub(r"[\*\#\-\_\<\>\/]+", "", response.text).strip()
        except Exception as e:
            return {
                "error": str(e),
                "timings": {"total_seconds": time.time() - run_start}
            }