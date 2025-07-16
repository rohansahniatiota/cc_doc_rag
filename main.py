from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI(title="PDF File Retrieval API", description="API to fetch PDF files from S3 buckets", version="1.0.0")

class Document(BaseModel):
    fileName: str
    url: str

class DocumentResponse(BaseModel):
    documents: List[Document]

@app.get("/documents", response_model=DocumentResponse)
async def get_pdf_documents():
    """
    Retrieve metadata for predefined PDF files from S3 buckets.
    """
    documents = [
        Document(
            fileName="Amazon Service Practice",
            url="https://d1.awsstatic.com/whitepapers/AmazonS3BestPractices.pdf"
        ),
        Document(
            fileName="Amazon Service",
            url="https://d1.awsstatic.com/whitepapers/AmazonS3BestPractices.pdf"
        )
    ]
    return {"documents": documents}
