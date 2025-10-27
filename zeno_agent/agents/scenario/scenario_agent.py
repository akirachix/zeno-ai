import os
import time
import re
import pandas as pd
from typing import Dict, Any

from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_crop_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    query_rag_embeddings_semantic
)

_SCEN_CACHE: Dict[str, tuple] = {}
_SCEN_TTL = 600 

def _scache_get(key: str):
    entry = _SCEN_CACHE.get(key)
    if entry and (time.time() - entry[0] < _SCEN_TTL):
        return entry[1]
    _SCEN_CACHE.pop(key, None)
    return None

def _scache_set(key: str, val: Any):
    _SCEN_CACHE[key] = (time.time(), val)

def embedding_reasoning_fallback(commodity: str, country: str, direction: str, pct: int, scenario_query: str) -> str:
    rag_articles = query_rag_embeddings_semantic(scenario_query, top_k=3)
    if rag_articles:
        lines = []
        for idx, article in enumerate(rag_articles, 1):
            context = article.get("content") or str(article)
            snippet = (context[:180] + "...") if len(context) > 180 else context
            lines.append(f"- Evidence {idx}: {snippet}")
        summary = "\n".join(lines)
        reasoning = "I used unstructured reports from Zeno's RAG DB to reason about this scenario."
        return summary + "\n\n" + reasoning + "\n\n" + "Source: Zeno RAG DB"
    else:
        return (
            f"I couldn't find relevant documents. Based on regional economic literature, "
            f"a {pct}% {direction} in {commodity} in {country.capitalize()} would likely affect prices and trade flows."
        )

class ScenarioSubAgent:
    def handle(self, scenario_query: str) -> Dict[str, Any]:
        q = scenario_query.lower()
        commodity_match = re.search(r"(maize|coffee|tea)", q)
        commodity = commodity_match.group(1) if commodity_match else None
        country_match = re.search(r"(kenya|uganda|tanzania|ethiopia|rwanda)", q)
        country = country_match.group(1) if country_match else "kenya"

        if "drop" in q or "decrease" in q:
            direction = "decrease"
            pct_match = re.search(r"(?:drop|decrease)(?: by)? (\d+)%", q)
            pct = int(pct_match.group(1)) if pct_match else 15
        elif "increase" in q or "raise" in q:
            direction = "increase"
            pct_match = re.search(r"(?:increase|raise)(?: by)? (\d+)%", q)
            pct = int(pct_match.group(1)) if pct_match else 15
        else:
            direction = None
            pct = 0

        if not commodity or not direction:
            out = {
                "response": "Try: 'What if maize price drops by 20% in Kenya over the next 3 months?'",
                "followup": "Specify commodity and direction."
            }
            return out

        key = f"scenario::{country}:{commodity}:{direction}:{pct}"
        cached = _scache_get(key)
        if cached:
            return cached

        try:
            country_id = get_country_id_by_name(country)
            product_id = get_crop_id_by_name(commodity) 
            indicator_id = get_indicator_id_by_metric("price")

            df = get_trade_data_from_db(
                country_id=country_id,
                product_id=product_id,
                indicator_id=indicator_id,
                start_date="2023-01-01"  
            )

            if df.empty or df['price'].isnull().all():
                raise ValueError("No price data available")

            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df = df.dropna(subset=['date', 'price'])
            df = df.sort_values('date').tail(6)

            base_prices = df['price'].tolist()
            base_months = df['date'].dt.strftime('%Y-%m').tolist()

            if not base_prices:
                raise ValueError("No valid price data")

            if direction == "decrease":
                scenario_prices = [round(p * (1 - pct / 100), 2) for p in base_prices]
            else:
                scenario_prices = [round(p * (1 + pct / 100), 2) for p in base_prices]

            explanation = (
                f"In {country.title()}, a {pct}% {direction} in {commodity} "
                f"would change monthly prices from {base_prices} to {scenario_prices}."
            )
            result = {
                "response": explanation,
                "graph_path": None,
                "followup": "Try another scenario?"
            }

        except Exception as e:
            fallback_text = embedding_reasoning_fallback(commodity, country, direction, pct, scenario_query)
            result = {
                "response": fallback_text,
                "followup": "Try a different scenario or check data availability.",
                "error": str(e)
            }

        _scache_set(key, result)
        return result