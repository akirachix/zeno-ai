import google.generativeai as genai
import os
import re
import json
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error
from .db_utils import (
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

class ForecastingAgent:
    def __init__(self):
        self.supported_commodities = ["maize", "coffee_arabica", "coffee_robusta", "tea"]
        self.supported_metrics = ["export_volume", "price", "revenue"]
        self.supported_countries = ["kenya", "ethiopia", "rwanda"]
        self.supported_models = ["ARIMA", "Prophet", "XGBoost", "Ensemble"]
        
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise EnvironmentError("GOOGLE_API_KEY environment variable is not set.")
        genai.configure(api_key=api_key)

    def validate_params(self, params: Dict[str, Any]) -> List[str]:
        """Validate input parameters against supported values."""
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
        """Parse timeframe string to number of months (e.g., 'next 2 years' → 24)."""
        match = re.match(r"next (\d+) (year|years|month|months)", timeframe.lower())
        if not match:
            raise ValueError(f"Invalid timeframe: {timeframe}")
        num = int(match.group(1))
        unit = match.group(2)
        return num * 12 if 'year' in unit else num

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess time-series data: handle missing values, outliers, and feature engineering."""
        df = df.copy()
        df['y'] = df['y'].interpolate(method='linear').ffill().bfill()
        df = df[np.abs(df['y'] - df['y'].mean()) <= (3 * df['y'].std())]
        df['lag_1'] = df['y'].shift(1)
        df['lag_12'] = df['y'].shift(12)
        df = df.dropna()
        return df.sort_values('ds').reset_index(drop=True)

    def select_model(self, df: pd.DataFrame, metric: str, periods: int) -> tuple[str, str]:
        """Select forecasting model based on data characteristics and return explanation."""
        freq = df['ds'].dt.inferred_freq
        has_seasonality = (freq in ['MS', 'M', 'QS', 'Q']) and len(df) >= 20
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
        """Calculate MAE, RMSE, and MAPE for forecast evaluation."""
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100 if not np.any(actual == 0) else float('inf')
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

    def run_prophet(self, df: pd.DataFrame, periods: int) -> tuple:
        """Run Prophet model for forecasting."""
        m = Prophet()
        m.fit(df[['ds', 'y']])
        future = m.make_future_dataframe(periods=periods, freq='MS')
        forecast_df = m.predict(future)
        forecast_series = forecast_df['yhat'].tail(periods).tolist()
        confidence_intervals = list(zip(forecast_df['yhat_lower'].tail(periods), forecast_df['yhat_upper'].tail(periods)))
        return forecast_series, confidence_intervals

    def run_arima(self, df: pd.DataFrame, periods: int) -> tuple:
        """Run ARIMA model for forecasting."""
        model = ARIMA(df['y'], order=(1,1,1))
        fitted = model.fit()
        forecast_result = fitted.get_forecast(steps=periods)
        forecast_series = forecast_result.predicted_mean.tolist()
        conf_int = forecast_result.conf_int(alpha=0.05)
        confidence_intervals = list(zip(conf_int['lower y'], conf_int['upper y']))
        return forecast_series, confidence_intervals

    def run_xgboost(self, df: pd.DataFrame, periods: int) -> tuple:
        """Run XGBoost model for forecasting."""
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
        confidence_intervals = [["Medium"] * periods] 
        return forecast_series, confidence_intervals

    def run_ensemble(self, df: pd.DataFrame, periods: int) -> tuple:
        """Run ensemble of ARIMA, Prophet, and XGBoost."""
        arima_forecast, _ = self.run_arima(df, periods)
        prophet_forecast, _ = self.run_prophet(df, periods)
        xgboost_forecast, _ = self.run_xgboost(df, periods)
        ensemble_forecast = np.mean([arima_forecast, prophet_forecast, xgboost_forecast], axis=0).tolist()
        confidence_intervals = [["Medium"] * periods]  
        return ensemble_forecast, confidence_intervals

    def calculate_confidence(self, confidence_intervals: list, std_dev: float) -> str:
        """Calculate confidence level based on interval width."""
        if not confidence_intervals or isinstance(confidence_intervals[0], str):
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
        """Adjust forecast based on RAG context (e.g., policy or weather events)."""
        for doc in rag_context:
            content = doc.get("content", "").lower()
            if "drought" in content:
                forecast = [x * 0.9 for x in forecast] 
            elif "policy change" in content:
                forecast = [x * 1.05 for x in forecast]  
        return forecast

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for forecasting.
        Uses historical data from zeno.trade_data and RAG context.
        Returns structured JSON with forecast, confidence, reasoning, sources.
        """
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
            if not has_structured_data:
                raise ValueError(f"No historical data found for {commodity} {metric} in {country}.")
            
            if 'month' in df.columns and df['month'].notna().any():
                df['ds'] = pd.to_datetime(df[['year', 'month']].assign(day=1))
            else:
                df['ds'] = pd.to_datetime(df['year'].astype(str) + '-01-01')
            
            df = df[['ds', 'value']].rename(columns={'value': 'y'}).dropna().sort_values('ds').reset_index(drop=True)
            if len(df) < 8:
                raise ValueError(f"Insufficient data points ({len(df)}) for reliable forecasting.")
            
            df = self.preprocess_data(df)
            print(f"Retrieved {len(df)} data points for {commodity} {metric} in {country}.")

            query_text = f"{original_commodity} {metric} {country} trend forecast {timeframe}"
            try:
                query_embedding = encode_query_to_vector(query_text)
                rag_context = query_rag_embeddings_semantic(query_embedding)
            except Exception as e:
                print(f"RAG retrieval failed: {e}")
                rag_context = [{"content": "RAG unavailable", "source": "N/A"}]
            
            if not rag_context:
                rag_context = [{"content": "No relevant documents found.", "source": "N/A"}]
            
            if run_id:
                log_step(run_id, step_order, "rag_retrieval", {
                    "query": query_text,
                    "num_results": len(rag_context),
                    "sources": [doc.get("source", "Unknown") for doc in rag_context]
                })
            step_order += 1

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
                test_forecast, _ = self.run_prophet(train_df, 8) if model_type == "Prophet" else self.run_arima(train_df, 8)
                metrics = self.evaluate_forecast(test_df['y'], test_forecast)
            else:
                metrics = {"MAE": "N/A", "RMSE": "N/A", "MAPE": "N/A"}
            
            commodity_assumption = (
                f"Assumed {commodity.replace('_', ' ').title()} for {country.title()} "
                f"as it is the dominant type based on regional agricultural data (e.g., KNBS reports)."
            ) if original_commodity != commodity else ""
            
            context_blocks = [f"- [{doc.get('source', 'Unknown')}] {doc.get('content', '')}" for doc in rag_context]
            context_str = "\n".join(context_blocks)
            prompt = f"""You are Zeno, an AI Economist Assistant for Kenyan Agricultural Trade.
Your task is to provide a clear, evidence-based explanation for a forecast generated from historical trade data.
User Query: "{original_commodity} {metric} in {country} for {timeframe}"
Numerical Forecast: {forecast_value}
Confidence Level: {confidence}
Model Used: {model_type}
Model Selection Reasoning: {model_explanation}
Evaluation Metrics: {json.dumps(metrics)}
Commodity Assumption: {commodity_assumption}
Retrieved Context: {context_str}
Instructions:
- Do NOT repeat the numerical forecast.
- Explain WHY the forecast makes sense based on retrieved documents.
- Mention key events: droughts, policy changes, global prices, infrastructure.
- If data is sparse, acknowledge limitations.
- Always cite sources.
- Include model selection reasoning and commodity assumption in the explanation.
- Format as plain text.
Example:
Assumed Arabica coffee for Kenya as it is the dominant type per KNBS reports. The forecast reflects rising global demand and improved post-harvest infrastructure, consistent with KNBS 2023 findings. Prophet was chosen due to detected seasonality in the data.
"""
            model = genai.GenerativeModel("gemini-1.5-pro")
            try:
                response = model.generate_content(prompt)
                reasoning = response.text.strip()
            except Exception as e:
                reasoning = f"LLM failed to generate explanation: {str(e)}"
            
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
            return {"error": str(e)}