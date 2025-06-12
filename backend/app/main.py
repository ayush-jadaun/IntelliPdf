from fastapi import FastAPI
from app.api import health


app = FastAPI()
app.include_router(health.router)


@app.get("/")
def read_root():
    return {"message": "FastAPI is working!"}
