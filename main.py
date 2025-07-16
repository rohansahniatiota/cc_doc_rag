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
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/1.%20Basics%20and%20overview.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/9.%20Deployment%20%26%20Managing%20Infrastructure%20at%20Scale.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/11.%20Cloud%20Integrations.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/12.%20Cloud%20Monitoring.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/13.%20VPC%20%26%20Networking.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/14.%20Security%20%26%20Compliance.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/16.%20Account%20Management%2C%20Billing%20%26%20Support.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/17.%20Advanced%20Identity.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/18.%20Other%20Services.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/19.%20AWS%20Architecting%20%26%20Ecosystem.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/2.%20IAM%20-%20Identity%20Access%20Management.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/20.%20Short%20Notes.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/3.%20EC2%20-%20Elastic%20Cloud%20Compute.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/4.%20EC2%20Storage%20and%20stuff.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/5.%20ELB%20%26%20ASG.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/7.%20Databases%20%26%20Analytics.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://github.com/tanishachandani/AWS-Notes/blob/main/8.%20Other%20Compute%20Services.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://drive.google.com/file/d/0B3vyNXp6qDWwTWdzLTJjOUtTcGc/edit?resourcekey=0-f1hRee21RT_lRLiIiequNw"
        ),
        Document(
            fileName="Deployment",
            url="https://www.emgywomenscollege.ac.in/templateEditor/kcfinder/upload/files/Encyclopaedia%20in%20Mathematics.pdf"
        ),
        Document(
            fileName="Deployment",
            url="https://darwin-online.org.uk/converted/pdf/1823_Encyclopedia_A770.06.pdf"
        )
        
        
    ]
    return {"documents": documents}
