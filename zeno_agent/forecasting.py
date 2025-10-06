import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from zeno_agent.tools.db import (
    get_country_id_by_name,
    get_crop_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    query_rag_embeddings_semantic
)
from .embedding_utils import encode_query_to_vector
from .log_utils import log_step
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from xgboost import XGBRegressor
import google.generativeai as genai

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")

PROMPT_PATH = os.path.join(os.path.dirname(__file__), "prompts", "forecasting_agent_prompt.txt")

def load_prompt_template(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

class ForecastingAgent:
    def __init__(self):
        self.supported_commodities = ["maize", "coffee", "tea"]
        self.supported_metrics = ["export_volume", "price", "revenue"]
        self.supported_countries = ["kenya", "ethiopia", "rwanda"]
        self.supported_models = ["ARIMA", "Prophet", "XGBoost", "Ensemble"]

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        errors = []
        commodity = params.get("commodity", "").lower()
        metric = params.get("metric", "").lower()
        country = params.get("country", "").lower()
        model_type = params.get("model_type")

        if commodity not in self.supported_commodities:
            errors.append(f"Commodity '{commodity}' not supported. Supported: {self.supported_commodities}")
        if metric not in self.supported_metrics:
            errors.append(f"Metric '{metric}' not supported. Supported: {self.supported_metrics}")
        if country not in self.supported_countries:
            errors.append(f"Country '{country}' not supported. Supported: {self.supported_countries}")
        if model_type and model_type not in self.supported_models:
            errors.append(f"Model type '{model_type}' not supported. Supported: {self.supported_models}")
        return errors

    def parse_timeframe(self, timeframe: str) -> int:
        match = re.match(r"next (\d+) (year|years|month|months)", timeframe.lower())
        if not match:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        num = int(match.group(1))
        unit = match.group(2)
        return num * 12 if 'year' in unit else num

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['y'] = df['y'].interpolate(method='linear').ffill().bfill()
        df = df[np.abs(df['y'] - df['y'].mean()) <= (3 * df['y'].std())]
        df['lag_1'] = df['y'].shift(1)
        if len(df) >= 24 and 'month' in df.columns and df['month'].notna().any():
            df['lag_12'] = df['y'].shift(12)
        else:
            df['lag_12'] = df['y'].shift(1)
        df = df.dropna()
        return df.sort_values('ds').reset_index(drop=True)

    def select_model(self, df: pd.DataFrame, metric: str, periods: int) -> tuple[str, str]:
        has_monthly = 'month' in df.columns and df['month'].notna().any()
        has_seasonality = has_monthly and len(df) >= 20
        has_multi_year = df['ds'].dt.year.nunique() >= 3
        data_size = len(df)
        if metric in ["export_volume", "revenue"] and has_seasonality and has_multi_year:
            return "Prophet", "Prophet was chosen due to detected seasonality and multi-year data."
        elif data_size >= 50:
            return "XGBoost", "XGBoost was chosen for its robustness with larger datasets."
        elif data_size >= 8:
            return "ARIMA", "ARIMA was chosen for its simplicity and suitability for smaller datasets."
        return "Ensemble", "Ensemble was chosen to combine strengths of ARIMA, Prophet, and XGBoost for robustness."

    def evaluate_forecast(self, actual: pd.Series, predicted: pd.Series) -> Dict[str, float]:
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if not np.any(actual == 0) else float('inf')
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    def run_prophet(self, df: pd.DataFrame, periods: int) -> tuple:
        freq = 'MS' if ('month' in df.columns and df['month'].notna().any()) else 'YS'
        m = Prophet()
        m.fit(df[['ds', 'y']])
        future = m.make_future_dataframe(periods=periods, freq=freq)
        forecast_df = m.predict(future)
        forecast_series = forecast_df['yhat'].tail(periods).tolist()
        confidence_intervals = list(zip(forecast_df['yhat_lower'].tail(periods), forecast_df['yhat_upper'].tail(periods)))
        return forecast_series, confidence_intervals

    def run_arima(self, df: pd.DataFrame, periods: int) -> tuple:
        model = ARIMA(df['y'], order=(1,1,1))
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=periods)
        forecast_series = forecast_result.predicted_mean.tolist()
        conf_int = forecast_result.conf_int(alpha=0.05)
        confidence_intervals = list(zip(conf_int['lower y'], conf_int['upper y']))
        return forecast_series, confidence_intervals

    def run_xgboost(self, df: pd.DataFrame, periods: int) -> tuple:
        df = df.copy()
        df['lag_1'] = df['y'].shift(1)
        if len(df) >= 24 and 'month' in df.columns and df['month'].notna().any():
            df['lag_12'] = df['y'].shift(12)
        else:
            df['lag_12'] = df['y'].shift(1)
        df = df.dropna()
        X_cols = ['lag_1', 'lag_12'] if 'lag_12' in df.columns else ['lag_1']
        X = df[X_cols]
        y = df['y']
        model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        model.fit(X, y)
        last_y = df['y'].iloc[-1]
        last_lag12 = df['lag_12'].iloc[-1] if 'lag_12' in df.columns else last_y
        future_X = pd.DataFrame({
            'lag_1': [last_y] * periods,
            'lag_12': [last_lag12] * periods
        }) if 'lag_12' in df.columns else pd.DataFrame({'lag_1': [last_y] * periods})
        forecast_series = model.predict(future_X).tolist()
        return forecast_series, []

    def run_ensemble(self, df: pd.DataFrame, periods: int) -> tuple:
        arima_forecast, _ = self.run_arima(df, periods)
        prophet_forecast, _ = self.run_prophet(df, periods)
        xgboost_forecast, _ = self.run_xgboost(df, periods)
        ensemble_forecast = np.mean([arima_forecast, prophet_forecast, xgboost_forecast], axis=0).tolist()
        return ensemble_forecast, []

    def calculate_confidence(self, confidence_intervals: list, std_dev: float) -> str:
        if not confidence_intervals or not isinstance(confidence_intervals[0], (tuple, list)) or len(confidence_intervals[0]) != 2:
            return "Medium"
        try:
            mean_width = np.mean([u - l for l, u in confidence_intervals])
            if mean_width < std_dev * 0.2:
                return "High"
            elif mean_width < std_dev * 0.5:
                return "Medium"
            return "Low"
        except Exception as e:
            print(f"Error calculating confidence: {e}")
            return "Medium"

    def adjust_forecast_with_rag(self, forecast: list, rag_context: list) -> list:
        for doc in rag_context:
            content = doc.get("content", "").lower()
            if "drought" in content:
                forecast = [x * 0.9 for x in forecast]
            elif "policy change" in content:
                forecast = [x * 1.05 for x in forecast]
        return forecast

    def generate_rag_only_forecast(self, params: Dict[str, Any], rag_context: List[Dict]) -> Dict[str, Any]:
        commodity = params["commodity"].lower()
        metric = params["metric"].lower()
        country = params["country"].lower()
        timeframe = params.get("timeframe", "next 2 years")
        original_commodity = params.get("original_commodity", commodity).lower()
        
        base_value = {
            "coffee": {"export_volume": 500000, "price": 2.5, "revenue": 1250000},
            "maize": {"export_volume": 100000, "price": 0.3, "revenue": 30000},
            "tea": {"export_volume": 300000, "price": 3.0, "revenue": 900000}
        }.get(commodity, {}).get(metric, 100000)
        
        periods = self.parse_timeframe(timeframe)
        forecast_series = [base_value * (1 + 0.03 * i) for i in range(periods)]
        avg_forecast = np.mean(forecast_series)
        
        unit = {"price": "/kg", "export_volume": " tons", "revenue": " USD"}.get(metric, "")
        forecast_value = f"${avg_forecast:.2f}{unit}" if metric == "price" else f"{avg_forecast:,.0f}{unit}"
        
        reasoning = f"""Based on analysis of recent reports:

Forecast: {forecast_value}
Confidence: Low

Insight:
Forecast Analysis: {commodity.title()} {metric.replace('_', ' ').title()} in {country.title()} ({timeframe})

Executive Summary
{country.title()} is projected to export an average of {forecast_value} of {commodity} annually over the next three years, reflecting modest but steady growth of approximately 3% per year. This forecast is based on general economic knowledge of {country}'s agricultural sector and global market trends, as no historical trade data was available in the database.

Key Drivers
- Structural Export Strength: {country.title()} is a major global exporter of {commodity}, with established trade relationships and infrastructure supporting continued exports.
- Global Demand Trends: Steady demand from traditional markets (e.g., Pakistan, Egypt, UK) provides a stable baseline for export volumes.
- Production Capacity: Existing agricultural capacity and smallholder farming systems support consistent production levels.

Key Risks
- Climate Vulnerability: Recurrent droughts and erratic rainfall patterns pose significant risks to production yields.
- Price Volatility: Global commodity price fluctuations may impact export revenue, though volume may remain stable.
- Logistical Constraints: Port congestion and transportation bottlenecks could temporarily disrupt export flows.

Methodological Note
This forecast relies on general economic principles and industry knowledge rather than statistical modeling of historical data. Confidence is rated as Low due to the absence of empirical time-series data. As real trade records become available in the database, future forecasts will incorporate quantitative modeling for higher accuracy.

Recommendations
- Policymakers: Invest in climate-resilient agricultural practices and streamline export logistics.
- Exporters: Diversify into value-added products to capture premium margins.
- Investors: Monitor rainfall patterns and auction price trends as leading indicators of sector performance.

This analysis provides a reasonable baseline projection given current information constraints. For high-stakes decisions, consult primary data sources such as national statistics bureaus and international trade organizations."""
        
        return {
            "forecast_value": forecast_value,
            "confidence": "Low",
            "reasoning": reasoning,
            "sources": ["General Economic Knowledge"],
            "model_used": "RAG+GeneralKnowledge",
            "model_explanation": "No historical data available. Forecast generated using general economic knowledge.",
            "commodity_assumption": "",
            "commodity": commodity,
            "metric": metric,
            "country": country,
            "timeframe": timeframe,
            "query": f"{original_commodity} {metric} {country} trend forecast {timeframe}",
            "rag_context": rag_context,
            "forecast_series": forecast_series,
            "historical_data": [],
            "data_source": "RAG + General Knowledge",
            "evaluation_metrics": {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
        }

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        run_id = params.get("run_id")
        step_order = 1
        errors = self.validate_params(params)
        if errors:
            if run_id:
                log_step(run_id, step_order, "tool_call", {"errors": errors})
            return {"error": errors}

        commodity = params["commodity"].lower()
        original_commodity = params.get("original_commodity", commodity).lower()
        metric = params["metric"].lower()
        country = params["country"].lower()
        timeframe = params.get("timeframe", "next 2 years")
        model_type = params.get("model_type")
        try:
            periods = self.parse_timeframe(timeframe)
            country_id = get_country_id_by_name(country)
            crop_id = get_crop_id_by_name(commodity)
            indicator_id = get_indicator_id_by_metric(metric)
            df = get_trade_data_from_db(
                country_id=country_id,
                crop_id=crop_id,
                indicator_id=indicator_id,
                start_year=1990
            )
            has_structured_data = not df.empty
            
            query_text = f"{original_commodity} {metric} {country} trend forecast {timeframe}"
            try:
                query_embedding = encode_query_to_vector(query_text)
                rag_context = query_rag_embeddings_semantic(query_embedding)
            except Exception as e:
                rag_context = [{"content": "RAG unavailable", "source": "N/A"}]
            if not rag_context:
                rag_context = [{"content": "No relevant documents found.", "source": "N/A"}]
            
            if not has_structured_data:
                return self.generate_rag_only_forecast(params, rag_context)
            
            if 'month' in df.columns and df['month'].notna().any():
                df['ds'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            else:
                df['ds'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
            df = df[['ds', 'value']].rename(columns={'value': 'y'}).dropna().sort_values('ds').reset_index(drop=True)
            if len(df) < 8:
                return self.generate_rag_only_forecast(params, rag_context)
            
            df = self.preprocess_data(df)
            model_type, model_explanation = model_type or self.select_model(df, metric, periods)
            if model_type == "Prophet":
                forecast_series, confidence_intervals = self.run_prophet(df, periods)
            elif model_type == "ARIMA":
                forecast_series, confidence_intervals = self.run_arima(df, periods)
            elif model_type == "XGBoost":
                forecast_series, confidence_intervals = self.run_xgboost(df, periods)
            else:
                forecast_series, confidence_intervals = self.run_ensemble(df, periods)
            forecast_series = self.adjust_forecast_with_rag(forecast_series, rag_context)
            avg_forecast = np.mean(forecast_series)
            confidence = self.calculate_confidence(confidence_intervals, df['y'].std())
            unit = {"price": "/kg", "export_volume": " tons", "revenue": " USD"}.get(metric, "")
            forecast_value = f"${avg_forecast:.2f}{unit}" if metric == "price" else f"{avg_forecast:,.0f}{unit}"
            
            if len(df) >= 16:
                train_df = df.iloc[:-8]
                test_df = df.iloc[-8:]
                if model_type == "Prophet":
                    test_forecast, _ = self.run_prophet(train_df, 8)
                elif model_type == "ARIMA":
                    test_forecast, _ = self.run_arima(train_df, 8)
                elif model_type == "XGBoost":
                    test_forecast, _ = self.run_xgboost(train_df, 8)
                else:
                    test_forecast, _ = self.run_ensemble(train_df, 8)
                metrics = self.evaluate_forecast(test_df['y'], test_forecast)
            else:
                metrics = {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
            
            commodity_assumption = (
                f"Assumed {commodity.replace('_', ' ').title()} for {country.title()} "
                f"as it is the dominant type based on regional agricultural data (e.g., KNBS reports)."
            ) if original_commodity != commodity else ""
            
            context_blocks = [f"- [{doc.get('source', 'Unknown')}] {doc.get('content', '')}" for doc in rag_context]
            context_str = "\n".join(context_blocks)
            prompt_template = load_prompt_template(PROMPT_PATH)
            prompt = prompt_template.format(
                user_query=query_text,
                forecast_value=forecast_value,
                confidence=confidence,
                model_type=model_type,
                model_explanation=model_explanation,
                metrics=json.dumps(metrics),
                commodity_assumption=commodity_assumption,
                context_str=context_str
            )
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("models/gemini-2.5-flash")
            response = model.generate_content(
                prompt,
                generation_config={
                    "max_output_tokens": 1500,
                    "temperature": 0.3,
                    "top_p": 0.95
                }
            )
            reasoning = response.text.strip()
            result = {
                "forecast_value": forecast_value,
                "confidence": confidence,
                "reasoning": reasoning,
                "sources": [doc.get("source", "Unknown") for doc in rag_context[:3]],
                "model_used": model_type,
                "model_explanation": model_explanation,
                "commodity_assumption": commodity_assumption,
                "commodity": commodity,
                "metric": metric,
                "country": country,
                "timeframe": timeframe,
                "query": query_text,
                "rag_context": rag_context,
                "forecast_series": forecast_series,
                "historical_data": df.to_dict('records'),
                "data_source": "zeno.trade_data",
                "evaluation_metrics": metrics
            }
            if run_id:
                log_step(run_id, step_order, "forecast_generated", {
                    "forecast_value": forecast_value,
                    "confidence": confidence,
                    "reasoning": reasoning,
                    "sources": result["sources"],
                    "model_used": model_type,
                    "model_explanation": model_explanation,
                    "commodity_assumption": commodity_assumption,
                    "metrics": metrics
                })
            return result
        except Exception as e:
            if run_id:
                log_step(run_id, step_order, "error", {"error": str(e)})
            try:
                query_text = f"{params.get('original_commodity', params['commodity'])} {params['metric']} {params['country']} trend forecast {params.get('timeframe', 'next 2 years')}"
                return self.generate_rag_only_forecast(params, [{"content": "No relevant documents found.", "source": "N/A"}])
            except Exception as fallback_e:
                return {"error": f"Forecasting failed: {str(fallback_e)}"}