import os
import psycopg2
from psycopg2.pool import SimpleConnectionPool
from psycopg2.extensions import register_adapter, AsIs
import numpy as np
import pandas as pd
from typing import Optional
from cachetools import TTLCache

def encode_vector_for_postgres(vector: list[float]) -> str:
    if not isinstance(vector, list) or not all(isinstance(x, (float, int)) for x in vector):
        raise ValueError("Input must be a list of floats/ints.")
    return "[" + ",".join(f"{v:.12f}" for v in vector) + "]"


db_pool = None
cache = TTLCache(maxsize=1000, ttl=3600)

def adapt_vector(vec):
    if isinstance(vec, np.ndarray):
        vec = vec.tolist()
    return AsIs("'" + ",".join(str(x) for x in vec) + "'")

register_adapter(np.ndarray, adapt_vector)
register_adapter(list, adapt_vector)

def init_db_pool():
    global db_pool
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        raise EnvironmentError("DATABASE_URL environment variable is not set.")
    db_pool = SimpleConnectionPool(1, 20, db_url)

def get_db_connection():
    global db_pool
    if db_pool is None:
        init_db_pool()
    return db_pool.getconn()

def release_db_connection(conn):
    global db_pool
    if db_pool and conn is not None:
        try:
            db_pool.putconn(conn)
        except Exception as e:
            print(f"Error releasing connection: {e}")

def get_country_id_by_name(country_name: str) -> int:
    cache_key = f"country_{country_name.lower()}"
    if cache_key in cache:
        return cache[cache_key]
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT id FROM zeno.countries WHERE LOWER(name) = LOWER(%s)", (country_name.strip(),))
        result = cur.fetchone()
        if not result:
            raise ValueError(f"Country '{country_name}' not found.")
        cache[cache_key] = result[0]
        return result[0]
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)

def get_crop_id_by_name(crop_name: str) -> int:
    cache_key = f"crop_{crop_name.lower()}"
    if cache_key in cache:
        return cache[cache_key]

    conn = None  
    try:
        conn = get_db_connection()  
        cur = conn.cursor()
        
        cur.execute("SELECT id FROM zeno.products WHERE LOWER(name) = LOWER(%s)", (crop_name.strip(),))
        result = cur.fetchone()
        if result:
            cache[cache_key] = result[0]
            return result[0]
            
        crop_lower = crop_name.lower()
        if "coffee" in crop_lower:
            coffee_names = ["Coffee", "Arabica Coffee", "Coffee Arabica", "coffee"]
            for name in coffee_names:
                cur.execute("SELECT id FROM zeno.products WHERE LOWER(name) = LOWER(%s)", (name,))
                result = cur.fetchone()
                if result:
                    cache[cache_key] = result[0]
                    return result[0]
                    
        cur.execute("SELECT id FROM zeno.products WHERE LOWER(name) LIKE %s", (f"%{crop_name.strip().lower()}%",))
        result = cur.fetchone()
        if result:
            cache[cache_key] = result[0]
            return result[0]

        raise ValueError(f"Crop '{crop_name}' not found in zeno.products.")
    finally:
        if conn is not None:  
            cur.close()
            release_db_connection(conn)

def get_indicator_id_by_metric(metric: str) -> int:
    cache_key = f"metric_{metric.lower()}"
    if cache_key in cache:
        return cache[cache_key]

    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        if metric.lower() in ["export_volume", "export volume"]:
            export_names = ["exports", "export", "export volume", "export_volume", "total exports"]
            for name in export_names:
                cur.execute("SELECT id FROM zeno.indicators WHERE LOWER(name) = %s", (name,))
                result = cur.fetchone()
                if result:
                    cache[cache_key] = result[0]
                    return result[0]
        
        elif metric.lower() == "price":
            cur.execute("SELECT id FROM zeno.indicators WHERE LOWER(name) = 'price'")
            result = cur.fetchone()
            if result:
                cache[cache_key] = result[0]
                return result[0]
                
        elif metric.lower() == "revenue":
            cur.execute("SELECT id FROM zeno.indicators WHERE LOWER(name) = 'revenue'")
            result = cur.fetchone()
            if result:
                cache[cache_key] = result[0]
                return result[0]

        cur.execute("SELECT name FROM zeno.indicators")
        all_indicators = [row[0] for row in cur.fetchall()]
        raise ValueError(f"Metric '{metric}' not found. Available: {all_indicators}")
        
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)

def get_trade_data_from_db(country_id: int, product_id: int, indicator_id: int, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            SELECT td.date, td.quantity, td.price, td.currency, u.name as quantity_unit_name, u.symbol as quantity_unit_symbol, td.source, td.metadata
            FROM zeno.trade_data td
            LEFT JOIN zeno.units u ON td.unit_id = u.id
            WHERE td.country_id = %s AND td.product_id = %s AND td.indicator_id = %s
        """
        params = [country_id, product_id, indicator_id]
        if start_date:
            query += " AND td.date >= %s"
            params.append(start_date)
        if end_date:
            query += " AND td.date <= %s"
            params.append(end_date)
        query += " ORDER BY td.date ASC"
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = ["date", "quantity", "price", "currency", "quantity_unit_name", "quantity_unit_symbol", "source", "metadata"]
        return pd.DataFrame(rows, columns=columns)
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)

def get_macro_stats_from_db(country_id: int, indicator_id: int, start_year: Optional[int] = None, end_year: Optional[int] = None) -> pd.DataFrame:
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        query = """
            SELECT ms.year, ms.value, ms.source, ms.metadata
            FROM zeno.macro_stats ms
            WHERE ms.country_id = %s AND ms.indicator_id = %s
        """
        params = [country_id, indicator_id]
        if start_year:
            query += " AND ms.year >= %s"
            params.append(start_year)
        if end_year:
            query += " AND ms.year <= %s"
            params.append(end_year)
        query += " ORDER BY ms.year ASC"
        cur.execute(query, params)
        rows = cur.fetchall()
        columns = ["year", "value", "source", "metadata"]
        return pd.DataFrame(rows, columns=columns)
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)

def query_rag_embeddings_semantic(query_embedding, top_k: int = 10):
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        vector_str = encode_vector_for_postgres(query_embedding)
        cur.execute("""
            SELECT content, source
            FROM zeno.rag_embeddings
            ORDER BY embedding_vector <-> %s::vector
            LIMIT %s
        """, (vector_str, top_k))
        results = cur.fetchall()
        return [{"content": r[0], "source": r[1]} for r in results]
    except Exception as e:
        print(f"[Warning] Semantic search failed: {e}")
        return []
    finally:
        if conn is not None:
            cur.close()
            release_db_connection(conn)

