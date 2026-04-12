from typing import Optional

from pydantic import BaseModel, Field

class RAGUsedContext(BaseModel):
    image_url: str = Field(..., description="The URL of the image of the item")
    ## Price can be null on the items stored in qdrant db
    price: Optional[float] = Field(None, description="The price of the item")
    description: str = Field(..., description="The description of the item")


class RAGRequest(BaseModel):
    query: str = Field(..., description="Query to be used in the RAG pipeline")

class RAGResponse(BaseModel):
    request_id: str
    answer: str = Field(..., description="Answer to be used in the RAG pipeline")
    used_context:list[RAGUsedContext] = Field(..., description="Information about the items used to answer the query")