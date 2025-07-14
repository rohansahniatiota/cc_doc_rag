# from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, HttpUrl
# from typing import Optional, Dict, Any
# import asyncio
# import aiohttp
# import aiofiles
# import os
# import uuid
# from datetime import datetime
# import json
# from urllib.parse import urlparse, parse_qs
# import re
# from pathlib import Path
# import logging
# from contextlib import asynccontextmanager

# # Import cloud service handlers
# from google.oauth2 import service_account
# from googleapiclient.discovery import build
# from googleapiclient.http import MediaIoBaseDownload
# import io
# import requests
# from msal import ConfidentialClientApplication
# # import dropbox
# # from boxsdk import OAuth2, Client
# import base64

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Configuration
# DOWNLOAD_BASE_PATH = os.getenv("DOWNLOAD_PATH", "./downloads")
# CALLBACK_URL = os.getenv("CALLBACK_URL", "")  # Optional callback URL

# # Cloud service configurations
# GOOGLE_SERVICE_ACCOUNT_FILE = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "service-account.json")
# # MICROSOFT_CLIENT_ID = os.getenv("MICROSOFT_CLIENT_ID", "")
# # MICROSOFT_CLIENT_SECRET = os.getenv("MICROSOFT_CLIENT_SECRET", "")
# # MICROSOFT_TENANT_ID = os.getenv("MICROSOFT_TENANT_ID", "")
# # DROPBOX_ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN", "")
# # BOX_CLIENT_ID = os.getenv("BOX_CLIENT_ID", "")
# # BOX_CLIENT_SECRET = os.getenv("BOX_CLIENT_SECRET", "")
# # BOX_ACCESS_TOKEN = os.getenv("BOX_ACCESS_TOKEN", "")

# # In-memory storage for job status (use Redis in production)
# job_status = {}

# class FetchRequest(BaseModel):
#     folder_url: HttpUrl
#     job_id: Optional[str] = None
#     callback_url: Optional[str] = None

# class FetchResponse(BaseModel):
#     job_id: str
#     status: str
#     message: str

# class JobStatus(BaseModel):
#     job_id: str
#     status: str
#     progress: Dict[str, Any]
#     created_at: datetime
#     completed_at: Optional[datetime] = None
#     error: Optional[str] = None

# class CloudFileManager:
#     def __init__(self):
#         self.download_path = DOWNLOAD_BASE_PATH
#         Path(self.download_path).mkdir(exist_ok=True)

#     def detect_service(self, url: str) -> str:
#         """Detect cloud service from URL"""
#         url_lower = url.lower()
#         if 'drive.google.com' in url_lower:
#             return 'gdrive'
#         # elif 'onedrive' in url_lower or '1drv.ms' in url_lower or 'sharepoint' in url_lower:
#         #     return 'onedrive'
#         # elif 'box.com' in url_lower:
#         #     return 'box'
#         # elif 'dropbox.com' in url_lower:
#         #     return 'dropbox'
#         else:
#             raise ValueError(f"Unsupported cloud service: {url}")

#     async def fetch_files(self, job_id: str, folder_url: str, callback_url: str = None): # type: ignore
#         """Main method to fetch files from cloud storage"""
#         try:
#             service = self.detect_service(folder_url)
#             job_status[job_id]["status"] = "processing"
#             job_status[job_id]["progress"]["service"] = service
            
#             if service == 'gdrive':
#                 await self.fetch_gdrive_files(job_id, folder_url)
#             # elif service == 'onedrive':
#             #     await self.fetch_onedrive_files(job_id, folder_url)
#             # elif service == 'box':
#             #     await self.fetch_box_files(job_id, folder_url)
#             # elif service == 'dropbox':
#             #     await self.fetch_dropbox_files(job_id, folder_url)
            
#             job_status[job_id]["status"] = "completed"
#             job_status[job_id]["completed_at"] = datetime.now()
#             job_status[job_id]["progress"]["message"] = "All files fetched successfully"
            
#             if callback_url:
#                 await self.send_callback(callback_url, job_id, "success")
                
#         except Exception as e:
#             logger.error(f"Error fetching files for job {job_id}: {str(e)}")
#             job_status[job_id]["status"] = "failed"
#             job_status[job_id]["error"] = str(e)
#             job_status[job_id]["completed_at"] = datetime.now()
            
#             if callback_url:
#                 await self.send_callback(callback_url, job_id, "error", str(e))

#     async def fetch_gdrive_files(self, job_id: str, folder_url: str):
#         """Fetch files from Google Drive"""
#         try:
#             # Extract folder ID from URL
#             folder_id = self.extract_gdrive_folder_id(folder_url)
            
#             # Authenticate
#             credentials = service_account.Credentials.from_service_account_file(
#                 GOOGLE_SERVICE_ACCOUNT_FILE,
#                 scopes=['https://www.googleapis.com/auth/drive']
#             )
#             service = build('drive', 'v3', credentials=credentials)
            
#             # Get folder contents
#             results = service.files().list(
#                 q=f"'{folder_id}' in parents",
#                 fields="files(id, name, mimeType, size)"
#             ).execute()
            
#             files = results.get('files', [])
#             job_status[job_id]["progress"]["total_files"] = len(files)
#             job_status[job_id]["progress"]["downloaded_files"] = 0
            
#             job_folder = os.path.join(self.download_path, job_id)
#             os.makedirs(job_folder, exist_ok=True)
            
#             for i, file in enumerate(files):
#                 try:
#                     # Download file
#                     request = service.files().get_media(fileId=file['id'])
#                     fh = io.BytesIO()
#                     downloader = MediaIoBaseDownload(fh, request)
                    
#                     done = False
#                     while not done:
#                         status, done = downloader.next_chunk()
                    
#                     # Save file
#                     file_path = os.path.join(job_folder, file['name'])
#                     async with aiofiles.open(file_path, 'wb') as f:
#                         await f.write(fh.getvalue())
                    
#                     job_status[job_id]["progress"]["downloaded_files"] = i + 1
                    
#                 except Exception as e:
#                     logger.error(f"Error downloading file {file['name']}: {str(e)}")
#                     continue
                    
#         except Exception as e:
#             raise Exception(f"Google Drive error: {str(e)}")

#     # async def fetch_onedrive_files(self, job_id: str, folder_url: str):
#     #     """Fetch files from OneDrive"""
#     #     try:
#     #         # Get access token
#     #         app = ConfidentialClientApplication(
#     #             client_id=MICROSOFT_CLIENT_ID,
#     #             client_credential=MICROSOFT_CLIENT_SECRET,
#     #             authority=f"https://login.microsoftonline.com/{MICROSOFT_TENANT_ID}"
#     #         )
            
#     #         result = app.acquire_token_for_client(
#     #             scopes=["https://graph.microsoft.com/.default"]
#     #         )
            
#     #         if "access_token" not in result: # type: ignore
#     #             raise Exception("Could not acquire access token")
            
#     #         headers = {'Authorization': f'Bearer {result["access_token"]}'} # type: ignore
            
#     #         # Encode share URL
#     #         encoded_url = base64.urlsafe_b64encode(folder_url.encode()).decode().rstrip('=')
#     #         share_token = f"s!{encoded_url}"
            
#     #         # Get folder contents
#     #         async with aiohttp.ClientSession() as session:
#     #             async with session.get(
#     #                 f"https://graph.microsoft.com/v1.0/shares/{share_token}/driveItem/children",
#     #                 headers=headers
#     #             ) as response:
#     #                 if response.status != 200:
#     #                     raise Exception(f"Failed to access OneDrive folder: {response.status}")
                    
#     #                 data = await response.json()
#     #                 files = data.get('value', [])
                    
#     #                 job_status[job_id]["progress"]["total_files"] = len(files)
#     #                 job_status[job_id]["progress"]["downloaded_files"] = 0
                    
#     #                 job_folder = os.path.join(self.download_path, job_id)
#     #                 os.makedirs(job_folder, exist_ok=True)
                    
#     #                 for i, file in enumerate(files):
#     #                     if file.get('file'):  # It's a file, not a folder
#     #                         try:
#     #                             async with session.get(
#     #                                 file['@microsoft.graph.downloadUrl'],
#     #                                 headers=headers
#     #                             ) as download_response:
#     #                                 if download_response.status == 200:
#     #                                     content = await download_response.read()
#     #                                     file_path = os.path.join(job_folder, file['name'])
#     #                                     async with aiofiles.open(file_path, 'wb') as f:
#     #                                         await f.write(content)
                                        
#     #                                     job_status[job_id]["progress"]["downloaded_files"] = i + 1
                                        
#     #                         except Exception as e:
#     #                             logger.error(f"Error downloading file {file['name']}: {str(e)}")
#     #                             continue
                                
#     #     except Exception as e:
#     #         raise Exception(f"OneDrive error: {str(e)}")

#     # async def fetch_dropbox_files(self, job_id: str, folder_url: str):
#     #     """Fetch files from Dropbox"""
#     #     try:
#     #         dbx = dropbox.Dropbox(DROPBOX_ACCESS_TOKEN)
            
#     #         # Get shared folder metadata
#     #         shared_folder_metadata = dbx.sharing_get_shared_link_metadata(folder_url)
            
#     #         # List files in shared folder
#     #         result = dbx.files_list_folder(shared_folder_metadata.path_lower)
            
#     #         files = [entry for entry in result.entries if isinstance(entry, dropbox.files.FileMetadata)] # type: ignore
#     #         job_status[job_id]["progress"]["total_files"] = len(files)
#     #         job_status[job_id]["progress"]["downloaded_files"] = 0
            
#     #         job_folder = os.path.join(self.download_path, job_id)
#     #         os.makedirs(job_folder, exist_ok=True)
            
#     #         for i, file in enumerate(files):
#     #             try:
#     #                 _, response = dbx.files_download(file.path_lower)
                    
#     #                 file_path = os.path.join(job_folder, file.name)
#     #                 async with aiofiles.open(file_path, 'wb') as f:
#     #                     await f.write(response.content)
                    
#     #                 job_status[job_id]["progress"]["downloaded_files"] = i + 1
                    
#     #             except Exception as e:
#     #                 logger.error(f"Error downloading file {file.name}: {str(e)}")
#     #                 continue
                    
#     #     except Exception as e:
#     #         raise Exception(f"Dropbox error: {str(e)}")

#     # async def fetch_box_files(self, job_id: str, folder_url: str):
#     #     """Fetch files from Box"""
#     #     try:
#     #         oauth = OAuth2(
#     #             client_id=BOX_CLIENT_ID,
#     #             client_secret=BOX_CLIENT_SECRET,
#     #             access_token=BOX_ACCESS_TOKEN
#     #         )
#     #         client = Client(oauth)
            
#     #         # Extract folder ID from URL
#     #         folder_id = self.extract_box_folder_id(folder_url)
            
#     #         # Get folder contents
#     #         folder = client.folder(folder_id)
#     #         items = folder.get_items()
            
#     #         files = [item for item in items if item.type == 'file']
#     #         job_status[job_id]["progress"]["total_files"] = len(files)
#     #         job_status[job_id]["progress"]["downloaded_files"] = 0
            
#     #         job_folder = os.path.join(self.download_path, job_id)
#     #         os.makedirs(job_folder, exist_ok=True)
            
#     #         for i, file in enumerate(files):
#     #             try:
#     #                 file_path = os.path.join(job_folder, file.name)
#     #                 with open(file_path, 'wb') as f:
#     #                     file.download_to(f)
                    
#     #                 job_status[job_id]["progress"]["downloaded_files"] = i + 1
                    
#     #             except Exception as e:
#     #                 logger.error(f"Error downloading file {file.name}: {str(e)}")
#     #                 continue
                    
#     #     except Exception as e:
#     #         raise Exception(f"Box error: {str(e)}")

#     def extract_gdrive_folder_id(self, url: str) -> str:
#         """Extract folder ID from Google Drive URL"""
#         patterns = [
#             r'/folders/([a-zA-Z0-9-_]+)',
#             r'id=([a-zA-Z0-9-_]+)'
#         ]
        
#         for pattern in patterns:
#             match = re.search(pattern, url)
#             if match:
#                 return match.group(1)
        
#         raise ValueError("Could not extract Google Drive folder ID")

#     # def extract_box_folder_id(self, url: str) -> str:
#     #     """Extract folder ID from Box URL"""
#     #     match = re.search(r'/folder/(\d+)', url)
#     #     if match:
#     #         return match.group(1)
        
#     #     raise ValueError("Could not extract Box folder ID")

#     async def send_callback(self, callback_url: str, job_id: str, status: str, error: str = None): # type: ignore
#         """Send callback notification"""
#         try:
#             payload = {
#                 "job_id": job_id,
#                 "status": status,
#                 "timestamp": datetime.now().isoformat(),
#                 "error": error
#             }
            
#             async with aiohttp.ClientSession() as session:
#                 async with session.post(callback_url, json=payload) as response:
#                     logger.info(f"Callback sent for job {job_id}: {response.status}")
                    
#         except Exception as e:
#             logger.error(f"Error sending callback: {str(e)}")

# # Initialize FastAPI app
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     # Startup
#     logger.info("Starting Cloud File Fetcher API")
#     yield
#     # Shutdown
#     logger.info("Shutting down Cloud File Fetcher API")

# app = FastAPI(
#     title="Cloud File Fetcher API",
#     description="API to fetch files from cloud storage services for GPT Actions",
#     version="1.0.0",
#     lifespan=lifespan
# )

# # CORS middleware
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Initialize cloud file manager
# cloud_manager = CloudFileManager()

# @app.get("/")
# async def root():
#     return {"message": "Cloud File Fetcher API is running"}

# @app.post("/fetch-files", response_model=FetchResponse)
# async def fetch_files(request: FetchRequest, background_tasks: BackgroundTasks):
#     """
#     Start fetching files from a cloud storage folder
#     """
#     job_id = request.job_id or str(uuid.uuid4())
    
#     # Initialize job status
#     job_status[job_id] = {
#         "job_id": job_id,
#         "status": "started",
#         "progress": {
#             "total_files": 0,
#             "downloaded_files": 0,
#             "service": None,
#             "message": "Job started"
#         },
#         "created_at": datetime.now(),
#         "completed_at": None,
#         "error": None
#     }
    
#     # Add background task
#     background_tasks.add_task(
#         cloud_manager.fetch_files,
#         job_id,
#         str(request.folder_url),
#         request.callback_url # type: ignore
#     )
    
#     return FetchResponse(
#         job_id=job_id,
#         status="started",
#         message="File fetching job started successfully"
#     )

# @app.get("/job-status/{job_id}", response_model=JobStatus)
# async def get_job_status(job_id: str):
#     """
#     Get the status of a file fetching job
#     """
#     if job_id not in job_status:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     return JobStatus(**job_status[job_id])

# @app.get("/list-jobs")
# async def list_jobs():
#     """
#     List all jobs and their statuses
#     """
#     return {"jobs": list(job_status.keys()), "total": len(job_status)}

# @app.delete("/job/{job_id}")
# async def delete_job(job_id: str):
#     """
#     Delete a job and its downloaded files
#     """
#     if job_id not in job_status:
#         raise HTTPException(status_code=404, detail="Job not found")
    
#     # Delete downloaded files
#     job_folder = os.path.join(DOWNLOAD_BASE_PATH, job_id)
#     if os.path.exists(job_folder):
#         import shutil
#         shutil.rmtree(job_folder)
    
#     # Remove from job status
#     del job_status[job_id]
    
#     return {"message": f"Job {job_id} deleted successfully"}

# @app.get("/health")
# async def health_check():
#     """
#     Health check endpoint
#     """
#     return {
#         "status": "healthy",
#         "timestamp": datetime.now().isoformat(),
#         "active_jobs": len([j for j in job_status.values() if j["status"] in ["started", "processing"]])
#     }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




# initialize fastapi app
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
import requests
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Initialize FastAPI app
app = FastAPI(
    title="S3 PDF RAG API",
    description="API to fetch PDF from S3, store in Chroma vector DB, and retrieve context for RAG",
    version="1.0.0"
)

# Set up Chroma vector store (persistent)
CHROMA_DIR = "./chroma_db"
os.makedirs(CHROMA_DIR, exist_ok=True)
EMBEDDINGS = OpenAIEmbeddings()
VECTOR_DB = Chroma(persist_directory=CHROMA_DIR, embedding_function=EMBEDDINGS)

# Pydantic models
class S3PDFIngestRequest(BaseModel):
    s3_url: str
    doc_id: Optional[str] = None  # Optional identifier for the document

class RAGQueryRequest(BaseModel):
    query: str
    top_k: int = 3
    doc_id: Optional[str] = None  # If provided, restrict search to this doc

class RAGQueryResponse(BaseModel):
    query: str
    results: List[dict]  # Each dict: {"content": ..., "metadata": ...}

@app.post("/ingest-pdf")
async def ingest_pdf(req: S3PDFIngestRequest):
    """
    Download a PDF from an S3 link, split, embed, and store in Chroma vector DB.
    """
    # Download PDF to temp file
    try:
        response = requests.get(req.s3_url)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to download PDF: {e}")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(response.content)
        tmp_pdf_path = tmp_file.name

    # Load and split PDF
    try:
        loader = PyPDFLoader(tmp_pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        # Add doc_id to metadata if provided
        if req.doc_id:
            for s in splits:
                s.metadata["doc_id"] = req.doc_id
    except Exception as e:
        os.remove(tmp_pdf_path)
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {e}")

    # Store in Chroma
    try:
        VECTOR_DB.add_documents(splits)
        VECTOR_DB.persist()
    except Exception as e:
        os.remove(tmp_pdf_path)
        raise HTTPException(status_code=500, detail=f"Failed to store in vector DB: {e}")

    os.remove(tmp_pdf_path)
    return {"message": "PDF ingested and stored successfully", "num_chunks": len(splits), "doc_id": req.doc_id}

@app.post("/rag-query", response_model=RAGQueryResponse)
async def rag_query(req: RAGQueryRequest):
    """
    Query the vector DB for relevant context chunks (RAG style).
    """
    # Build filter if doc_id is provided
    filter_kwargs = {}
    if req.doc_id:
        filter_kwargs = {"doc_id": req.doc_id}

    try:
        docs_and_scores = VECTOR_DB.similarity_search_with_score(req.query, k=req.top_k, filter=filter_kwargs if filter_kwargs else None)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vector DB query failed: {e}")

    results = []
    for doc, score in docs_and_scores:
        results.append({
            "content": doc.page_content,
            "metadata": doc.metadata,
            "score": score
        })

    return RAGQueryResponse(query=req.query, results=results)

@app.get("/list-docs")
async def list_docs():
    """
    List all unique doc_ids in the vector DB.
    """
    # Chroma doesn't have a direct API for this, so we scan all metadata
    all = VECTOR_DB.get(include=["metadatas"])
    print(all)
    doc_ids = set()
    for meta in all["metadatas"]:
        if meta and "doc_id" in meta:
            doc_ids.add(meta["doc_id"])
    return {"doc_ids": list(doc_ids)}



# deploy this app on render.com
