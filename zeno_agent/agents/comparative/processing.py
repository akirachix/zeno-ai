import pandas as pd
from typing import Dict, Any
from .utils import (
    client,
    extract_entities,
    calculate_cagr,
    merge_rag_content,
    encode_query_to_vector,
)
from zeno_agent.db_utils import (
    get_country_id_by_name,
    get_product_id_by_name,
    get_indicator_id_by_metric,
    get_trade_data_from_db,
    query_rag_embeddings_semantic,
)


def summarize_trade_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Return summarized metrics for trade data without exposing raw data."""
    if df.empty:
        return {"summary": "No trade data available.", "cagr": 0.0}

    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df = df.dropna(subset=["quantity", "price"])
    if df.empty:
        return {"summary": "No trade data available.", "cagr": 0.0}

    yearly = df.groupby("year").agg({"quantity": "sum", "price": "mean"}).reset_index()
    start_year, end_year = yearly["year"].min(), yearly["year"].max()
    total_quantity = yearly["quantity"].sum()
    avg_price = yearly["price"].mean()

    if len(yearly) > 1:
        cagr = calculate_cagr(
            float(yearly.iloc[0]["quantity"]), float(yearly.iloc[-1]["quantity"]), end_year - start_year
        )
    else:
        cagr = 0.0

    return {
        "summary": f"Period {start_year}-{end_year}: total {total_quantity:,} units, "
                   f"avg price KES {avg_price:,.2f}, CAGR {cagr:.2f}%",
        "cagr": cagr,
    }


def get_structured_summary(query: str) -> Dict[str, str]:
    entities = extract_entities(query)
    countries = entities["countries"]
    commodity = entities["commodity"]
    results = {}

    for country in countries:
        try:
            country_id = get_country_id_by_name(country)
            product_id = get_product_id_by_name(commodity)
            indicator_id = get_indicator_id_by_metric("exports")
            df = get_trade_data_from_db(country_id, product_id, indicator_id)
            summary = summarize_trade_data(df)
            results[country] = summary["summary"]
        except Exception as e:
            results[country] = f"{country}: Data unavailable ({e})"
    return results


def get_rag_evidence(query: str, top_k: int = 5) -> str:
    try:
        embedding = encode_query_to_vector(query)
        raw_results = query_rag_embeddings_semantic(embedding, top_k=top_k)
        return merge_rag_content(raw_results) if raw_results else ""
    except Exception:
        return ""


def synthesize_comparative_analysis(query: str, structured_data: Dict[str, str], rag_text: str) -> str:
    """Generate safe comparative report without exposing internal data."""
    context_lines = [f"{country}: {summary}" for country, summary in structured_data.items()]
    context = "\n".join(context_lines)
    if rag_text:
        context += f"\n\nContextual evidence from reports:\n{rag_text}"

    prompt = f"""
You are an economist producing a comparative report.
User Query: "{query}"

Available summarized trade metrics:
{context}

Guidelines:
- Use only summarized metrics; do not reveal raw numeric tables or data points.
- Focus on relative performance, trends, competitiveness, and growth.
- Write a professional economic narrative with plain text only (no markdown).
- Include insights and policy implications concisely.
"""
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text.strip()
