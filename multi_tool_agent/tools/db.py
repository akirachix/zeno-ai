import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from datetime import datetime, timedelta

import numpy as np

import google.generativeai as genai

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set! Please ensure .env exists in your project root and contains a valid DATABASE_URL line.")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set! Please ensure .env exists in your project root and contains a valid GOOGLE_API_KEY line.")

engine = create_engine(DATABASE_URL)
genai.configure(api_key=GOOGLE_API_KEY)

def get_text_embedding(text):
    """Returns a list embedding for a string using Gemini Embeddings API."""
    res = genai.embed_content(
        model="models/text-embedding-004",
        content=text,
        task_type="retrieval_document"
    )
    return res["embedding"]

def get_trade_data(commodity, country, last_n_months=6, return_raw=False):
    now = datetime.now()
    start_date = now - timedelta(days=last_n_months * 30)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT year, month, price, volume
                    FROM trade_data
                    WHERE lower(commodity) = :commodity
                      AND lower(country) = :country
                      AND (date_trunc('month', make_date(year, month, 1)) >= :startdate)
                    ORDER BY year, month
                """),
                {
                    "commodity": commodity.lower(),
                    "country": country.lower(),
                    "startdate": start_date.date()
                }
            )
            rows = result.fetchall()
            months = [f"{row.month}/{row.year}" for row in rows]
            prices = [float(row.price) for row in rows]

            try:
                meta_result = conn.execute(
                    text("""
                        SELECT source, updated_at, notes
                        FROM trade_data_metadata
                        WHERE lower(commodity) = :commodity
                          AND lower(country) = :country
                        LIMIT 1
                    """),
                    {
                        "commodity": commodity.lower(),
                        "country": country.lower(),
                    }
                )
                meta_row = meta_result.fetchone()
                metadata = None
                if meta_row:
                    metadata = dict(meta_row._mapping) if hasattr(meta_row, "_mapping") else dict(meta_row)
            except Exception as meta_e:
                print(f"DB warning: metadata query failed: {meta_e}")
                metadata = None

            if return_raw:
                return {"months": months, "prices": prices, "rows": rows, "metadata": metadata}
            return {"months": months, "prices": prices, "metadata": metadata}
    except Exception as e:
        print(f"DB error in get_trade_data: {e}")
        return {"months": [], "prices": [], "metadata": None}

def get_trade_data_by_year(commodity, country, start_year, end_year):
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT year, month, price, volume
                    FROM trade_data
                    WHERE lower(commodity) = :commodity
                      AND lower(country) = :country
                      AND year BETWEEN :start_year AND :end_year
                    ORDER BY year, month
                """),
                {
                    "commodity": commodity.lower(),
                    "country": country.lower(),
                    "start_year": start_year,
                    "end_year": end_year,
                }
            )
            rows = result.fetchall()
            months = [f"{row.month}/{row.year}" for row in rows]
            prices = [float(row.price) for row in rows]
            return {"months": months, "prices": prices}
    except Exception as e:
        print(f"DB error in get_trade_data_by_year: {e}")
        return {"months": [], "prices": []}

def semantic_search_rag_embeddings(user_query, top_k=5):
    """
    Semantic search for similar RAG embedding rows using pgvector.
    Returns top_k most similar rows by cosine distance.
    Assumes zeno.rag_embeddings table has an 'embedding_vector' column of type vector(768).
    """
    query_embedding = get_text_embedding(user_query)
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT embedding_id, content, source, embedding_vector,
                           (embedding_vector <#> (:query_embedding::vector)) AS cosine_distance
                    FROM zeno.rag_embeddings
                    ORDER BY cosine_distance ASC
                    LIMIT :top_k
                """),
                {
                    "query_embedding": query_embedding,
                    "top_k": top_k
                }
            )
            rows = result.fetchall()
            return [dict(row._mapping) if hasattr(row, "_mapping") else dict(row) for row in rows]
    except Exception as e:
        print(f"DB error in semantic_search_rag_embeddings: {e}")
        return []