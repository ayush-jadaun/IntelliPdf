from sqlalchemy import Column, Integer, String
from pgvector.sqlalchemy import Vector
from app.core.database.session import Base

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String)
    embedding = Column(Vector(768))  # Adjust vector size if needed
