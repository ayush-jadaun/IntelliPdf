from fastapi import APIRouter
from .endpoints import documents, search, chat, graph, insights

api_router = APIRouter()
api_router.include_router(documents.router, prefix="/documents", tags=["Documents"])
api_router.include_router(search.router, prefix="/search", tags=["Search"])
api_router.include_router(chat.router, prefix="/chat", tags=["Chat"])
api_router.include_router(graph.router, prefix="/graph", tags=["Graph"])
api_router.include_router(insights.router, prefix="/insights", tags=["Insights"])