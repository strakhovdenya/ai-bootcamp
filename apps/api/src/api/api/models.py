from pydantic import BaseModel, Field


class RAGRequest(BaseModel):
    query: str = Field(..., description="Query to be used in the RAG pipeline")

class RAGResponse(BaseModel):
    request_id: str
    answer: str = Field(..., description="Answer to be used in the RAG pipeline")