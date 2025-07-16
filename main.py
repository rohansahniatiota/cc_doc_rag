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
            fileName="wa2490.supplement.aws-s3.pdf",
            url="https://d33idl3etu5qjr.cloudfront.net/acorn/documents/wa2490.supplement.aws-s3.pdf"
        ),
        Document(
            fileName="s3-gsg.pdf",
            url="https://s3.amazonaws.com/awsdocs/S3/latest/s3-gsg.pdf"
        )
    ]
    return {"documents": documents}
