import psycopg2
from psycopg2.pool import SimpleConnectionPool
import os
import pandas as pd
from typing import Optional
from cachetools import TTLCache

db_pool = None
cache = TTLCache(maxsize=1000, ttl=3600)

def init_db_pool():
    global db_pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL environment variable is not set.")
    db_pool = SimpleConnectionPool(1, 5, db_url)

def get_db_connection():
    global db_pool
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()

def release_db_connection(conn):
    global db_pool
    db_pool.putconn(conn)

def get_country_id_by_name(country_name: str) -> int:
    cache_key = f"country_{country_name.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM zeno.countries WHERE LOWER(name) = LOWER(%s)",
            (country_name.strip(),)
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Country '{country_name}' not found in zeno.countries.")
        cache[cache_key] = result[0]
        return result[0]
    finally:
        cur.close()
        release_db_connection(conn)

def get_crop_id_by_name(commodity: str) -> int:
    cache_key = f"crop_{commodity.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM zeno.products WHERE LOWER(name) = LOWER(%s)",
            (commodity.strip(),)
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Commodity '{commodity}' not found in zeno.products.")
        cache[cache_key] = result[0]
        return result[0]
    finally:
        cur.close()
        release_db_connection(conn)

def get_indicator_id_by_metric(metric: str) -> int:
    cache_key = f"metric_{metric.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM zeno.indicators WHERE LOWER(name) = %s",
            (metric.lower(),)
        )
        result = cur.fetchone()
        if result:
            cache[cache_key] = result[0]
            return result[0]
        
        indicator_mapping = {'price': 1, 'export_volume': 2, 'revenue': 3}
        if metric.lower() in indicator_mapping:
            indicator_id = indicator_mapping[metric.lower()]
            cur.execute("SELECT id FROM zeno.indicators WHERE id = %s", (indicator_id,))
            if cur.fetchone():
                cache[cache_key] = indicator_id
                return indicator_id
        
        raise ValueError(f"Metric '{metric}' not found in zeno.indicators.")
    finally:
        cur.close()
        release_db_connection(conn)

def get_trade_data_from_db(
    country_id: int,
    crop_id: int,
    indicator_id: int,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None
) -> pd.DataFrame:
    conn = get_db_connection()
    cur = conn.cursor()
    
    value_column = "price" if indicator_id == 1 else "quantity"
    
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
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = ['year', 'month', 'value', 'source', 'metadata']
        df = pd.DataFrame(rows, columns=columns)
        return df
    finally:
        cur.close()
        release_db_connection(conn)

def query_rag_embeddings_semantic(query_embedding, top_k=5):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT content, source FROM zeno.rag_embeddings
            ORDER BY embedding_vector <-> %s::vector
            LIMIT %s
            """,
            (query_embedding, top_k)
        )
        results = cur.fetchall()
        return [{"content": r[0], "source": r[1]} for r in results]
    except Exception as e:
        print(f"Semantic search failed: {e}")
        return []
    finally:
        cur.close()
        release_db_connection(conn)