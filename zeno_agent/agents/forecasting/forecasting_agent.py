import re, json, numpy as np
from .config import client, ALIAS_MAP, SUPPORTED_COMMODITIES, SUPPORTED_COUNTRIES
from .data_utils import convert_to_usd, prepare_dual_data, get_enhanced_rag_context
from .model_utils import run_model
from zeno_agent.db_utils import get_country_id_by_name, get_product_id_by_name


class ForecastingAgent:
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
            interpretation = f"Interpretation unavailable: {e}"

        return {
            "type": "forecast",
            "query": query,
            "forecast_display": display_text,
            "dual_forecast": dual_forecast,
            "interpretation": interpretation,
            "confidence_level": "High" if len(df) >= 24 else "Medium",
            "data_points_used": len(df),
        }
