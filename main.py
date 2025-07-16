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
            fileName="Amazon S3",
            url="https://d33idl3etu5qjr.cloudfront.net/acorn/documents/wa2490.supplement.aws-s3.pdf"
        ),
        Document(
            fileName="AmazonS3BestPractices",
            url="https://d1.awsstatic.com/whitepapers/AmazonS3BestPractices.pdf"
        ),
        Document(
            fileName="Amazon-S3-Backup-for-Dummies-Clumio",
            url="https://prodgatsby.clumio.com/wp-content/uploads/2023/03/Amazon-S3-Backup-for-Dummies-Clumio.pdf"
        ),
        Document(
            fileName="AWS Simple Storage",
            url="https://awsdocs.s3.amazonaws.com/S3/latest/s3-api.pdf"
        ),
        Document(
            fileName="Global Infrastructure",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/10.%20AWS%20Global%20Infrastructure.pdf"
        ),
        Document(
            fileName="Machine Learning",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/15.%20Machine%20Learning.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/9.%20Deployment%20%26%20Managing%20Infrastructure%20at%20Scale.pdf"
        ),
        
        
    ]
    return {"documents": documents}
