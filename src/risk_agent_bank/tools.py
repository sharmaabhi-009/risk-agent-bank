import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional
from risk_agent_bank import send_llm_request
import re
import mysql.connector
import os
import json
from google.adk.tools import ToolContext
from solace_ai_connector.common.log import log
from solace_agent_mesh.agent.utils.artifact_helpers import (
    save_artifact_with_metadata,
    DEFAULT_SCHEMA_MAX_KEYS,
)
from solace_agent_mesh.agent.utils.context_helpers import get_original_session_id


HISTORY_LIMIT = int(os.getenv("HISTORY_LIMIT", "5"))  # fallback limit

MYSQL_CONFIG = {
    "host": "localhost",
    "user": "bank",
    "password": "bank",
    "database": "bank_db",
    "port": 3306,
}

def fetch_customer_data(customer_id: int):
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cur = conn.cursor(dictionary=True)

        cur.execute("SELECT * FROM customer_profile WHERE customer_id=%s", (customer_id,))
        profile = cur.fetchone()

        cur.execute(
            "SELECT * FROM login_history WHERE customer_id=%s ORDER BY login_time DESC LIMIT %s",
            (customer_id, HISTORY_LIMIT)
        )
        logins = cur.fetchall()

        cur.execute(
            "SELECT * FROM transactions WHERE customer_id=%s ORDER BY transaction_date DESC LIMIT %s",
            (customer_id, HISTORY_LIMIT * 2)
        )
        transactions = cur.fetchall()

        cur.close()
        conn.close()

        return {
            "profile": profile,
            "recent_logins": logins,
            "recent_transactions": transactions,
        }
    except Exception as e:
        log.error(f"[fetch_customer_data] DB error: {e}", exc_info=True)
        return {"error": str(e)}
    
    


async def fun_assess_fraud_risk(
    user_msg: str,
    tool_context: Optional[ToolContext] = None,
    tool_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Assess the fraud/risk associated with the banking transaction

    Args:
        user_msg: The natural language question/query
        tool_context: SAM framework context (provided automatically)
        tool_config: Tool-specific configuration (from config.yaml)

    Returns:
        A dictionary with status and the generated SQL query string
    """
    match = re.search(r"\b(\d+)\b", user_msg)
    log_identifier = "[fun_assess_fraud_risk]"
    if not match:
        log.error(f"{log_identifier} No customer_id found in input: {user_msg}")
        return {
            "status": "error",
            "error": "No customer_id found in input",
            "history": None,
            "response": None
        }
    
    log.info(f"{log_identifier} Processing NLP input: {user_msg}")
    customer_id = int(match.group(1))
    log.info(f"[fun_assess_fraud_risk] Fetching data for customer_id={customer_id}")

        # Fetch customer data from DB
    data = fetch_customer_data(customer_id)
    log.info(f"data fetched for customer_id=={data}")
    
    
    system_prompt = (
    "You are a banking fraud detection assistant. Fetch customer history from tables in JSON format and return the risk for the customer "
    "with keys: risk_score (0-100), "
    "anomaly_reasons (list of strings), recommendation (ALLOW/REVIEW/BLOCK)."
    )

    user_prompt = (
    f"Customer History: {data}\n"
    f"Transaction Details: {user_msg}\n"
    "Risk Assessment:"
    )
    
    final_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    llm_response=await send_llm_request(tool_context,final_prompt)
 

    return {
        "status": "success",
        "history": data,
        
        "response": llm_response,
    }
