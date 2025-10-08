
import psycopg2
import os
from datetime import datetime
from typing import Optional, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from .db_utils import get_db_connection, release_db_connection

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def log_run(conversation_id: int, user_input: str, final_output: str, status: str) -> Optional[int]:
    """Log a run to zeno.runs table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO zeno.runs (conversation_id, user_input, final_output, status, started_at, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s) RETURNING run_id
            """, (conversation_id, user_input, final_output, status, datetime.now(), datetime.now()))
            run_id = cur.fetchone()[0]
            conn.commit()
            return run_id
    except Exception as e:
        print(f"Error logging run: {e}")
        raise
    finally:
        release_db_connection(conn)

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def log_step(
    run_id: int,
    step_order: int,
    type: str,
    content: Any,
    tool_id: Optional[int] = None,
    agent_id: Optional[int] = None
) -> None:
    """Log a step to zeno.steps table."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO zeno.steps (run_id, step_order, type, content, tool_id, agent_id, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, (run_id, step_order, type, str(content), tool_id, agent_id, datetime.now()))
            conn.commit()
    except Exception as e:
        print(f"Error logging step: {e}")
        raise
    finally:
        release_db_connection(conn)
