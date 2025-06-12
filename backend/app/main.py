from fastapi import FastAPI
from app.api import health
from app.api.v1 import api

app = FastAPI()
app.include_router(health.router)
app.include_router(api.api_router)


@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}
