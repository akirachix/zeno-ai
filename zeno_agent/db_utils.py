import psycopg2
from psycopg2.pool import SimpleConnectionPool
import os
import pandas as pd
from typing import Optional
from cachetools import TTLCache


db_pool = None
cache = TTLCache(maxsize=1000, ttl=3600) 


def init_db_pool():
    """Initialize database connection pool."""
    global db_pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL environment variable is not set.")
    db_pool = SimpleConnectionPool(1, 20, db_url)


def get_db_connection():
    """Get a connection from the pool."""
    global db_pool
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()


def release_db_connection(conn):
    """Release a connection back to the pool."""
    global db_pool
    db_pool.putconn(conn)


def get_country_id_by_name(country_name: str) -> int:
    """Fetch country_id from zeno.countries by name."""
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
    """Fetch crop_id from zeno.crops by name."""
    cache_key = f"crop_{commodity.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT id FROM zeno.crops WHERE LOWER(name) = LOWER(%s)",
            (commodity.strip(),)
        )
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Commodity '{commodity}' not found in zeno.crops.")
        cache[cache_key] = result[0]
        return result[0]
    finally:
        cur.close()
        release_db_connection(conn)


def get_indicator_id_by_metric(metric: str) -> int:
    """
    Fetch indicator_id from zeno.indicators by matching metric name.
    Assumes indicators have names like 'Gross Output (Agriculture)', 'Commodity Price', etc.
    Uses fuzzy matching for user-friendly inputs.
    """
    cache_key = f"metric_{metric.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        metric_mappings = {
            "export_volume": ["gross output (agriculture)", "export volume"],
            "price": ["commodity price", "price"],
            "revenue": ["value added (agriculture)", "revenue"]
        }
        possible_names = metric_mappings.get(metric.lower(), [metric.lower()])
        for name in possible_names:
            cur.execute(
                "SELECT id FROM zeno.indicators WHERE LOWER(name) LIKE %s",
                (f"%{name}%",)
            )
            result = cur.fetchone()
            if result:
                cache[cache_key] = result[0]
                return result[0]
        raise ValueError(
            f"Metric '{metric}' not found in zeno.indicators. "
            "Ensure the indicators table includes relevant names (e.g., 'Gross Output (Agriculture)', 'Commodity Price')."
        )
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
    """Fetch historical trade data from zeno.trade_data table."""
    conn = get_db_connection()
    cur = conn.cursor()
    query = """
        SELECT
            td.year,
            td.month,
            td.value,
            td.source,
            td.metadata
        FROM zeno.trade_data td
        WHERE td.country_id = %s
          AND td.product_id = %s
          AND td.indicator_id = %s
    """
    params = [country_id, crop_id, indicator_id]
    
    if start_year:
        query += " AND td.year >= %s"
        params.append(start_year)
    if end_year:
        query += " AND td.year <= %s"
        params.append(end_year)
    
    query += " ORDER BY td.year ASC, td.month ASC"
    
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
    """Perform semantic similarity search using pgvector on zeno.rag_embeddings."""
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