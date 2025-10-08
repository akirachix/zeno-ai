import re
import numpy as np
import pandas as pd
from typing import Tuple, List, Any
from .config import EXCHANGE_RATES
from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_product_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    query_rag_embeddings_semantic,
)
from zeno_agent.embedding_utils import encode_query_to_vector


def convert_to_usd(value: float, currency: str) -> float:
    rate = EXCHANGE_RATES.get(currency.upper(), 1.0)
    return value * rate


def prepare_dual_data(country_id: int, product_id: int) -> Tuple[pd.DataFrame, str, str, str]:
    indicator_id = get_indicator_id_by_metric("exports")
    df = get_trade_data_from_db(country_id, product_id, indicator_id)

    if df.empty:
        raise ValueError("No trade data found.")
    if "date" not in df.columns:
        raise ValueError("No date column found.")
    if "quantity" not in df.columns or "price" not in df.columns:
        raise ValueError("Both quantity and price columns required.")

    df["ds"] = pd.to_datetime(df["date"])
    currency = df.get("currency", pd.Series(["KES"])).iloc[0] or "KES"

    unit_name = df.get("quantity_unit_name", pd.Series(["tonnes"])).iloc[0]
    unit_symbol = df.get("quantity_unit_symbol", pd.Series(["t"])).iloc[0]

    unit_str = f"{unit_name or ''} {unit_symbol or ''}".lower()
    if "ton" in unit_str or unit_symbol == "t":
        kg_multiplier = 1000
        display_unit = "tonnes"
    elif "kg" in unit_str:
        kg_multiplier = 1
        display_unit = "kg"
    elif "quintal" in unit_str:
        kg_multiplier = 100
        display_unit = "quintals"
    else:
        kg_multiplier = 1000
        display_unit = "tonnes"

    df["quantity_kg"] = df["quantity"] * kg_multiplier
    df["unit_price"] = df["price"] / df["quantity_kg"].replace(0, np.nan)
    df_clean = df[["ds", "quantity_kg", "price", "unit_price"]].dropna().sort_values("ds")

    if len(df_clean) < 8:
        raise ValueError(f"Insufficient data points ({len(df_clean)}).")

    return df_clean, currency, display_unit, unit_symbol


def get_enhanced_rag_context(commodity: str, country: str, metric: str) -> str:
    queries = [
        f"{commodity} {metric} {country} policy subsidies tariffs regulations export restrictions",
        f"{commodity} global market trends 2024 2025 supply demand",
        f"{country} export partners {commodity} trade agreements",
        f"{commodity} production challenges {country} climate drought disease",
    ]
    all_context = []
    for query in queries:
        try:
            emb = encode_query_to_vector(query)
            results = query_rag_embeddings_semantic(emb, top_k=2)
            for r in results:
                content = r.get("content", "")
                if len(content) > 100 and content[0].isupper():
                    all_context.append(content)
        except Exception:
            continue
    if not all_context:
        return "No recent policy or trade context found."
    return " ".join(dict.fromkeys(all_context))[:2000]
