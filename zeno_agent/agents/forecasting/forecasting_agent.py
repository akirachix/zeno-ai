"""
East African Commodity Forecasting Agent

Provides time-series forecasts for commodity prices, volumes, and revenues
using statistical models (Prophet, ARIMA, Linear Trend) with market context
from RAG knowledge base.

FILE: zeno_agent/agents/forecasting/forecasting_agent.py
"""

import re
import json
import numpy as np
import pandas as pd
from datetime import datetime
from decimal import Decimal
from dateutil.relativedelta import relativedelta
from .config import client, ALIAS_MAP, SUPPORTED_COMMODITIES, SUPPORTED_COUNTRIES
from .data_utils import convert_to_usd, prepare_dual_data, get_enhanced_rag_context
from .model_utils import run_model
from zeno_agent.db_utils import get_country_id_by_name, get_product_id_by_name


class ForecastingAgent:
    """
    Forecasting agent for commodity price, volume, and revenue predictions.
    """
    
    @staticmethod
    def _convert_decimals_to_float(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert all Decimal columns to float to avoid type errors in calculations.
        
        Args:
            df: DataFrame potentially containing Decimal types
            
        Returns:
            DataFrame with all Decimal columns converted to float
        """
        for col in df.columns:
            if df[col].dtype == object:
                # Check if column contains Decimal objects
                if len(df) > 0 and isinstance(df[col].iloc[0], Decimal):
                    df[col] = df[col].astype(float)
        return df
    
    def extract_target_date(self, query: str) -> dict:
        """Extract the specific target date from the query."""
        query_lower = query.lower()
        
        # Extract year
        year_match = re.search(r'(202[4-9]|203[0-9])', query)
        target_year = int(year_match.group(1)) if year_match else datetime.now().year + 1
        
        # Extract month
        months = {
            'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
            'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6,
            'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
            'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
        }
        
        target_month = None
        for month_name, month_num in months.items():
            if month_name in query_lower:
                target_month = month_num
                break
        
        # If no month specified, use current month or next month
        if target_month is None:
            target_month = datetime.now().month
        
        target_date = datetime(target_year, target_month, 1)
        
        return {
            'target_date': target_date,
            'year': target_year,
            'month': target_month
        }

    def parse_timeframe(self, timeframe: str) -> int:
        """Parse timeframe from natural language."""
        match = re.match(r"next (\d+) (year|years|month|months)", timeframe.lower())
        if not match:
            return 3
        num, unit = int(match.group(1)), match.group(2)
        return num if "month" in unit else num * 12

    def calculate_periods_to_target(self, last_date: pd.Timestamp, target_date: datetime) -> int:
        """Calculate number of months from last data point to target date."""
        months_diff = (target_date.year - last_date.year) * 12 + target_date.month - last_date.month
        return max(1, months_diff)  # At least 1 month ahead

    def _generate_data_preview(self, df: pd.DataFrame, currency: str, vol_unit: str) -> str:
        """Generate a formatted preview of historical data for the prompt."""
        # Get last 12 data points or all if less than 12
        preview_df = df.tail(12).copy()
        
        # Calculate basic statistics
        avg_price = float(df['unit_price'].mean())
        min_price = float(df['unit_price'].min())
        max_price = float(df['unit_price'].max())
        recent_trend = "increasing" if float(df['unit_price'].iloc[-1]) > float(df['unit_price'].iloc[0]) else "decreasing"
        
        # Calculate volatility (coefficient of variation)
        price_std = float(df['unit_price'].std())
        price_cv = (price_std / avg_price * 100) if avg_price > 0 else 0
        
        # Format recent data points
        data_points = []
        for _, row in preview_df.iterrows():
            date_str = row['ds'].strftime('%b %Y')
            price_str = f"{float(row['unit_price']):.2f}"
            volume_str = f"{float(row['quantity_kg']):.0f}"
            data_points.append(f"  • {date_str}: {price_str} {currency}/kg (Volume: {volume_str} {vol_unit})")
        
        preview = f"""
Summary Statistics:
  • Average Unit Price: {avg_price:.2f} {currency}/kg
  • Price Range: {min_price:.2f} - {max_price:.2f} {currency}/kg
  • Overall Trend: {recent_trend}
  • Price Volatility: {price_cv:.1f}% (coefficient of variation)

Recent Historical Data (last {len(preview_df)} periods):
{chr(10).join(data_points)}
        """
        
        return preview.strip()
    
    def forecast_dual_metrics(self, df, periods: int, target_period_index: int = None):
        """
        Forecast metrics for a specific period.
        
        Args:
            df: Historical data
            periods: Total periods to forecast
            target_period_index: Which period to return (0-indexed). If None, return average.
        """
        # Convert Decimals to float before modeling
        df = self._convert_decimals_to_float(df.copy())
        
        unit_df = df[["ds", "unit_price"]].rename(columns={"unit_price": "y"})
        unit_forecast, unit_ints, unit_model = run_model(unit_df, periods, "unit_price")

        rev_df = df[["ds", "price"]].rename(columns={"price": "y"})
        rev_forecast, rev_ints, rev_model = run_model(rev_df, periods, "revenue")

        vol_df = df[["ds", "quantity_kg"]].rename(columns={"quantity_kg": "y"})
        vol_forecast, _, _ = run_model(vol_df, periods, "volume")

        # If target_period_index specified, return that specific period
        if target_period_index is not None and target_period_index < len(unit_forecast):
            idx = target_period_index
            return {
                "unit_price": {
                    "forecast": float(unit_forecast[idx]),
                    "intervals": (float(unit_ints[idx][0]), float(unit_ints[idx][1])) if idx < len(unit_ints) and unit_ints[idx][0] is not None else (None, None),
                    "model": unit_model
                },
                "total_revenue": {
                    "forecast": float(rev_forecast[idx]),
                    "intervals": (float(rev_ints[idx][0]), float(rev_ints[idx][1])) if idx < len(rev_ints) and rev_ints[idx][0] is not None else (None, None),
                    "model": rev_model
                },
                "volume_kg": float(vol_forecast[idx]),
            }
        
        # Otherwise return average (legacy behavior)
        return {
            "unit_price": {"forecast": float(np.mean(unit_forecast)), "intervals": unit_ints, "model": unit_model},
            "total_revenue": {"forecast": float(np.mean(rev_forecast)), "intervals": rev_ints, "model": rev_model},
            "volume_kg": float(np.mean(vol_forecast)),
        }

    def run(self, inputs):
        """
        Execute forecasting for a commodity and country.
        
        Args:
            inputs: Dictionary with 'query' and optional 'file_context'
            
        Returns:
            Dictionary containing forecast results and interpretation
        """
        query = inputs.get("query", "")
        file_context = inputs.get("file_context", "")
        
        if not query:
            return {"error": "No query provided."}

        q = query.lower()
        commodity = next((k for k in ALIAS_MAP if k in q), None)
        country = next((c for c in SUPPORTED_COUNTRIES if c in q), None)

        if not commodity or not country:
            return {"error": "Could not identify commodity or country."}

        # Extract target date from query
        date_info = self.extract_target_date(query)
        target_date = date_info['target_date']

        commodity = ALIAS_MAP[commodity]
        country_id = get_country_id_by_name(country.title())
        product_id = get_product_id_by_name(commodity)
        
        try:
            df, currency, vol_unit, symbol = prepare_dual_data(country_id, product_id)
            # Convert Decimals to float immediately after data retrieval
            df = self._convert_decimals_to_float(df)
        except ValueError as e:
            return {"error": str(e)}
        except Exception as e:
            return {"error": f"Data preparation failed: {str(e)}"}
        
        rag_context = get_enhanced_rag_context(commodity, country, "price")

        # Calculate how many periods to forecast to reach target date
        last_data_date = df['ds'].max()
        periods = self.calculate_periods_to_target(last_data_date, target_date)
        
        # Get forecast for the specific target period (last period in the forecast)
        target_period_index = periods - 1
        dual_forecast = self.forecast_dual_metrics(df, periods, target_period_index)

        # Format the target date nicely
        target_date_str = target_date.strftime("%B %Y")

        display_text = (
            f"Forecast for {target_date_str}: "
            f"Unit Price: {dual_forecast['unit_price']['forecast']:.2f} {currency}/kg | "
            f"Revenue: {dual_forecast['total_revenue']['forecast']:.0f} {currency} | "
            f"Volume: {dual_forecast['volume_kg']:.0f} {vol_unit}"
        )

        # Build context section conditionally
        context_section = f"Context: {rag_context}"
        if file_context:
            context_section += f"\nAdditional document context: {file_context}"

        # Get confidence intervals if available
        unit_interval = dual_forecast['unit_price'].get('intervals', (None, None))
        interval_text = ""
        if unit_interval[0] is not None and unit_interval[1] is not None:
            interval_text = f"\nPrice Range (80% confidence): {unit_interval[0]:.2f} - {unit_interval[1]:.2f} {currency}/kg"

        # Generate historical data preview for the prompt
        data_preview = self._generate_data_preview(df, currency, vol_unit)

        prompt = f"""You are a senior economist at an East African trade research institution. Provide a professional forecast interpretation based on rigorous statistical analysis.

=== FORECAST SPECIFICATION ===
Commodity: {commodity}
Country: {country}
Target Period: {target_date_str}
Forecasting Model: {dual_forecast['unit_price']['model']}

=== FORECAST RESULTS ===
Predicted Unit Price: {dual_forecast['unit_price']['forecast']:.2f} {currency}/kg{interval_text}
Projected Total Revenue: {dual_forecast['total_revenue']['forecast']:.0f} {currency}
Expected Volume: {dual_forecast['volume_kg']:.0f} {vol_unit}

=== DATA FOUNDATION ===
Historical Observations: {len(df)} data points
Data Coverage: Through {last_data_date.strftime('%B %Y')}
Forecast Horizon: {periods} months forward
Model Confidence: {'High (sufficient data, short horizon)' if len(df) >= 24 and periods <= 12 else 'Medium (adequate data or moderate horizon)' if periods <= 24 else 'Low (extended forecast horizon)'}

=== HISTORICAL TRENDS ANALYSIS ===
{data_preview}

=== MARKET & POLICY CONTEXT ===
{context_section}

=== ANALYTICAL FRAMEWORK ===
Your analysis must address the following in clear, professional paragraphs (no bullet points, lists, or markdown formatting):

Paragraph 1 - Forecast Overview & Historical Context:
State the {target_date_str} price forecast explicitly. Compare to recent historical prices from the data preview. Identify whether forecast represents continuation, acceleration, or reversal of recent trends. Reference specific historical price points to establish context.

Paragraph 2 - Technical Analysis & Model Interpretation:
Explain how the {dual_forecast['unit_price']['model']} model processes the historical data. Discuss observed patterns: trend direction, volatility level, seasonality. Identify which historical patterns most influence this forecast. Note any structural breaks or anomalies in the data that affect projections.

Paragraph 3 - Risk Assessment & Confidence Bounds:
Evaluate forecast reliability given the {periods}-month horizon. Discuss data quality implications ({len(df)} observations, coverage gaps if any). Interpret confidence intervals (if available) and their practical meaning. Identify key risk factors from market context that could cause deviations. Consider exogenous shocks (policy changes, weather, global market shifts).

Paragraph 4 - Strategic Implications & Recommendations:
Advise producers on production and marketing strategies. Guide traders on inventory and hedging decisions. Inform policymakers on potential interventions or market monitoring needs. Reference regional trade dynamics (COMESA, EAC) and bilateral relationships. Suggest complementary actions to mitigate downside risks or capitalize on opportunities.

Write in clear, flowing prose using precise numerical references. Maintain analytical objectivity and support all claims with data. Target audience: Policy analysts, commodity traders, agricultural economists.
"""

        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            interpretation = re.sub(r"[\*\#\-\_\<\>\/]+", "", response.text).strip()
        except Exception as e:
            interpretation = f"Interpretation unavailable: {e}"

        return {
            "type": "forecast",
            "query": query,
            "target_date": target_date_str,
            "periods_ahead": periods,
            "forecast_display": display_text,
            "dual_forecast": dual_forecast,
            "interpretation": interpretation,
            "confidence_level": "High" if len(df) >= 24 and periods <= 12 else "Medium" if periods <= 24 else "Low",
            "data_points_used": len(df),
            "last_data_date": last_data_date.strftime('%B %Y'),
        }