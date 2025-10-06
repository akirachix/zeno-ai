import os
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool
import pandas as pd

DATABASE_URL = os.getenv("DATABASE_URL")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set!")
if not GOOGLE_API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is not set!")

engine = create_engine(DATABASE_URL, poolclass=NullPool)

def get_text_embedding(text: str) -> Optional[List[float]]:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        res = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
        )
        return res["embedding"]
    except Exception:
        return None

def embed_text(text: str) -> Optional[List[float]]:
    return get_text_embedding(text)

def get_country_id_by_name(country_name: str) -> int:
    country_mapping = {'kenya': 1, 'uganda': 2, 'tanzania': 3, 'rwanda': 4, 'ethiopia': 5}
    return country_mapping.get(country_name.lower(), 1)

def get_crop_id_by_name(commodity: str) -> int:
    crop_mapping = {'maize': 1, 'coffee': 2, 'tea': 3}
    return crop_mapping.get(commodity.lower(), 1)

def get_indicator_id_by_metric(metric: str) -> int:
    indicator_mapping = {'price': 1, 'export_volume': 2, 'revenue': 3}
    return indicator_mapping.get(metric.lower(), 1)

def get_trade_data(commodity: str, country: str, last_n_months: int = 6, return_raw: bool = False) -> Dict[str, Any]:
    try:
        country_id = get_country_id_by_name(country)
        crop_id = get_crop_id_by_name(commodity)
        indicator_id = get_indicator_id_by_metric("price")
        
        now = datetime.now()
        start_date = now - timedelta(days=last_n_months * 30)
        start_year = start_date.year
        start_month = start_date.month
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT EXTRACT(YEAR FROM date) as year, EXTRACT(MONTH FROM date) as month, price
                    FROM zeno.trade_data
                    WHERE country_id = :country_id
                      AND product_id = :product_id
                      AND indicator_id = :indicator_id
                      AND (EXTRACT(YEAR FROM date) > :start_year OR (EXTRACT(YEAR FROM date) = :start_year AND EXTRACT(MONTH FROM date) >= :start_month))
                    ORDER BY date
                """),
                {
                    "country_id": country_id,
                    "product_id": crop_id,
                    "indicator_id": indicator_id,
                    "start_year": start_year,
                    "start_month": start_month
                }
            )
            rows = result.fetchall()
            
            if not rows:
                try:
                    indicator_id_vol = get_indicator_id_by_metric("export_volume")
                    result = conn.execute(
                        text("""
                            SELECT EXTRACT(YEAR FROM date) as year, EXTRACT(MONTH FROM date) as month, quantity as price
                            FROM zeno.trade_data
                            WHERE country_id = :country_id
                              AND product_id = :product_id
                              AND indicator_id = :indicator_id
                              AND (EXTRACT(YEAR FROM date) > :start_year OR (EXTRACT(YEAR FROM date) = :start_year AND EXTRACT(MONTH FROM date) >= :start_month))
                            ORDER BY date
                        """),
                        {
                            "country_id": country_id,
                            "product_id": crop_id,
                            "indicator_id": indicator_id_vol,
                            "start_year": start_year,
                            "start_month": start_month
                        }
                    )
                    rows = result.fetchall()
                except:
                    pass
            
            months = [f"{int(row[1])}/{int(row[0])}" for row in rows if row[0] is not None and row[1] is not None]
            prices = [float(row[2]) if row[2] is not None else 0.0 for row in rows]

            try:
                meta_result = conn.execute(
                    text("""
                        SELECT source, updated_at
                        FROM zeno.trade_data
                        WHERE country_id = :country_id
                          AND product_id = :product_id
                          AND indicator_id = :indicator_id
                        LIMIT 1
                    """),
                    {
                        "country_id": country_id,
                        "product_id": crop_id,
                        "indicator_id": indicator_id,
                    }
                )
                meta_row = meta_result.fetchone()
                metadata = {"source": meta_row.source if meta_row else "Unknown", 
                           "updated_at": meta_row.updated_at if meta_row else None} if meta_row else None
            except:
                metadata = None

            if return_raw:
                return {"months": months, "prices": prices, "metadata": metadata}
            return {"months": months, "prices": prices, "metadata": metadata}
            
    except Exception:
        return {"months": [], "prices": [], "metadata": None}

def get_trade_data_by_year(commodity: str, country: str, start_year: int, end_year: int) -> Dict[str, Any]:
    try:
        country_id = get_country_id_by_name(country)
        crop_id = get_crop_id_by_name(commodity)
        indicator_id = get_indicator_id_by_metric("price")
        
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT EXTRACT(YEAR FROM date) as year, EXTRACT(MONTH FROM date) as month, price
                    FROM zeno.trade_data
                    WHERE country_id = :country_id
                      AND product_id = :product_id
                      AND indicator_id = :indicator_id
                      AND EXTRACT(YEAR FROM date) BETWEEN :start_year AND :end_year
                    ORDER BY date
                """),
                {
                    "country_id": country_id,
                    "product_id": crop_id,
                    "indicator_id": indicator_id,
                    "start_year": start_year,
                    "end_year": end_year,
                }
            )
            rows = result.fetchall()
            months = [f"{int(row[1])}/{int(row[0])}" for row in rows if row[0] is not None and row[1] is not None]
            prices = [float(row[2]) if row[2] is not None else 0.0 for row in rows]
            return {"months": months, "prices": prices}
    except:
        return {"months": [], "prices": []}

def semantic_search_rag_embeddings(user_query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_embedding = get_text_embedding(user_query)
    if query_embedding is None:
        return []
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT content, source
                    FROM zeno.rag_embeddings
                    ORDER BY embedding_vector <-> %s::vector
                    LIMIT %s
                """),
                (query_embedding, top_k)
            )
            rows = result.fetchall()
            return [{"content": row.content, "source": row.source} for row in rows]
    except Exception:
        return []

def query_embeddings(query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    query_vector = embed_text(query)
    if query_vector is None:
        return []
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT content, source, created_at
                    FROM zeno.rag_embeddings
                    ORDER BY embedding_vector <-> %s::vector
                    LIMIT %s
                """),
                (query_vector, top_k)
            )
            rows = result.fetchall()
            return [
                {
                    "content": row.content,
                    "source": row.source,
                    "created_at": str(row.created_at) if row.created_at else None
                }
                for row in rows
            ]
    except Exception:
        return [
            {"content": "Sample comparative data for East African coffee exports shows Kenya typically exports higher value Arabica coffee while Ethiopia focuses on volume with robusta varieties.", "source": "Mock Data"},
            {"content": "Maize trade patterns in East Africa show Uganda as a net exporter while Kenya often imports to meet domestic demand.", "source": "Mock Data"}
        ]

def get_trade_data_from_db(
    country_id: int,
    crop_id: int,
    indicator_id: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    if indicator_id == 1:
        value_column = "price"
    elif indicator_id == 2:
        value_column = "quantity"
    else:
        value_column = "quantity"
    
    query = f"""
        SELECT 
            EXTRACT(YEAR FROM td.date) as year,
            EXTRACT(MONTH FROM td.date) as month,
            td.{value_column} as value,
            td.source,
            td.metadata
        FROM zeno.trade_data td
        WHERE td.country_id = %s
          AND td.product_id = %s
          AND td.indicator_id = %s
          AND td.{value_column} IS NOT NULL
    """
    params = [country_id, crop_id, indicator_id]
    
    if start_year:
        query += " AND EXTRACT(YEAR FROM td.date) >= %s"
        params.append(start_year)
    if end_year:
        query += " AND EXTRACT(YEAR FROM td.date) <= %s"
        params.append(end_year)
    
    query += " ORDER BY td.date ASC"
    
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), params)
            rows = result.fetchall()
            columns = ['year', 'month', 'value', 'source', 'metadata']
            df = pd.DataFrame(rows, columns=columns)
            return df
    except Exception:
        return pd.DataFrame(columns=['year', 'month', 'value', 'source', 'metadata'])
    
def query_rag_embeddings_semantic(query_embedding, top_k=5):
    if not query_embedding:
        return []
    try:
        with engine.connect() as conn:
            result = conn.execute(
                text("""
                    SELECT content, source FROM zeno.rag_embeddings
                    ORDER BY embedding_vector <-> %s::vector
                    LIMIT %s
                """),
                (query_embedding, top_k)
            )
            rows = result.fetchall()
            return [{"content": r[0], "source": r[1]} for r in rows]
    except Exception:
        return []