# app/api/health.py
from fastapi import APIRouter
import psycopg2
import os

router = APIRouter()

@router.get("/health/db")
def check_db_health():
    try:
        conn = psycopg2.connect(
            dbname=os.getenv("POSTGRES_DB", "intellipdf_db"),
            user=os.getenv("POSTGRES_USER", "intellipdf"),
            password=os.getenv("POSTGRES_PASSWORD", "securepass"),
            host=os.getenv("POSTGRES_HOST", "postgres"),
            port=os.getenv("POSTGRES_PORT", "5432")
        )
        conn.close()
        return {"status": "ok", "message": "Connected to PostgreSQL!"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
