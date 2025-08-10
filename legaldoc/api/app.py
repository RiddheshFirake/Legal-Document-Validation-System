from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import yaml
import os
import uuid
import tempfile
import shutil
from pathlib import Path
import logging
import asyncio
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Initialize FastAPI app
app = FastAPI(
    title="Legal Document Validation API",
    description="AI-powered legal document legality detection system",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load configuration
config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize pipeline
from orchestrator.pipeline import DocumentValidationPipeline
pipeline = DocumentValidationPipeline(config)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class ValidationRequest(BaseModel):
    options: Optional[Dict[str, Any]] = {}
    return_detailed: bool = False
    async_processing: bool = False

class ValidationResponse(BaseModel):
    success: bool
    job_id: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

class BatchValidationRequest(BaseModel):
    options: Optional[Dict[str, Any]] = {}
    max_concurrent: int = 3
    return_detailed: bool = False

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "processing", "completed", "failed"
    progress: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime
    completed_at: Optional[datetime] = None

# In-memory job storage (in production, use Redis or database)
jobs = {}

# Helper functions
def save_uploaded_file(uploaded_file: UploadFile) -> str:
    """Save uploaded file to temporary location"""
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, uploaded_file.filename)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    
    return file_path

def cleanup_temp_file(file_path: str):
    """Clean up temporary file"""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
        # Also remove temp directory if empty
        temp_dir = os.path.dirname(file_path)
        if os.path.exists(temp_dir) and not os.listdir(temp_dir):
            os.rmdir(temp_dir)
    except Exception as e:
        logger.warning(f"Failed to cleanup temp file {file_path}: {e}")

async def process_document_async(job_id: str, file_path: str, options: Dict):
    """Process document asynchronously"""
    try:
        jobs[job_id]['status'] = 'processing'
        result = pipeline.process_document(file_path, options)
        result = convert_numpy_types(result)  # ADD THIS LINE
        
        jobs[job_id]['status'] = 'completed'
        jobs[job_id]['result'] = result
        jobs[job_id]['completed_at'] = datetime.now()
        
    except Exception as e:
        jobs[job_id]['status'] = 'failed'
        jobs[job_id]['error'] = str(e)
        jobs[job_id]['completed_at'] = datetime.now()
        logger.error(f"Async processing failed for job {job_id}: {e}")
    
    finally:
        cleanup_temp_file(file_path)


# API Routes

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Legal Document Validation API",
        "version": "1.0.0",
        "status": "active",
        "endpoints": {
            "validate": "/validate - Upload and validate a single document",
            "validate_batch": "/validate/batch - Upload and validate multiple documents",
            "status": "/status/{job_id} - Check job status for async processing",
            "health": "/health - API health check",
            "pipeline_status": "/pipeline/status - Check pipeline component status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        pipeline_status = pipeline.get_pipeline_status()
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "pipeline_status": pipeline_status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.get("/pipeline/status")
async def get_pipeline_status():
    """Get detailed pipeline component status"""
    try:
        status = pipeline.get_pipeline_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get pipeline status: {str(e)}")

@app.post("/validate", response_model=ValidationResponse)
async def validate_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    options: str = '{}',
    return_detailed: bool = False,
    async_processing: bool = False
):
    """Validate a single document"""
    
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Parse options
    try:
        import json
        options_dict = json.loads(options) if options else {}
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid options JSON")
    
    # Save uploaded file
    try:
        file_path = save_uploaded_file(file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")
    
    # Process asynchronously if requested
    if async_processing:
        job_id = str(uuid.uuid4())
        jobs[job_id] = {
            'status': 'pending',
            'created_at': datetime.now(),
            'file_name': file.filename
        }
        
        # Start background processing
        background_tasks.add_task(process_document_async, job_id, file_path, options_dict)
        
        return ValidationResponse(
            success=True,
            job_id=job_id,
            result=None
        )
    
    # Process synchronously
    try:
        result = pipeline.process_document(file_path, options_dict)
        result = convert_numpy_types(result)  # ADD THIS LINE to fix numpy serialization
        
        # Filter result based on return_detailed flag
        if not return_detailed and result.get('success'):
            filtered_result = {
                'success': result['success'],
                'decision': result.get('decision', {}),
                'processing_time': result.get('processing_time', 0)
            }
        else:
            filtered_result = result
        
        return ValidationResponse(
            success=result.get('success', False),
            result=filtered_result,
            processing_time=result.get('processing_time', 0)
        )
        
    except Exception as e:
        logger.error(f"Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
    
    finally:
        cleanup_temp_file(file_path)


@app.post("/validate/batch")
async def validate_documents_batch(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    options: str = '{}',
    max_concurrent: int = 3,
    return_detailed: bool = False
):
    """Validate multiple documents in batch"""
    
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")
    
    if len(files) > 10:  # Limit batch size
        raise HTTPException(status_code=400, detail="Too many files (max 10)")
    
    # Parse options
    try:
        import json
        options_dict = json.loads(options) if options else {}
        options_dict['max_concurrent'] = max_concurrent
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid options JSON")
    
    # Save all files
    file_paths = []
    try:
        for file in files:
            if file.filename:
                file_path = save_uploaded_file(file)
                file_paths.append(file_path)
    except Exception as e:
        # Cleanup any saved files
        for fp in file_paths:
            cleanup_temp_file(fp)
        raise HTTPException(status_code=500, detail=f"Failed to save files: {str(e)}")
    
    # Process batch
    try:
        results = await pipeline.process_batch(file_paths, options_dict)
        results = convert_numpy_types(results)
        
        # Filter results if not detailed
        if not return_detailed:
            filtered_results = []
            for result in results:
                if result.get('success'):
                    filtered_result = {
                        'success': result['success'],
                        'file_path': result.get('file_path'),
                        'decision': result.get('decision', {}),
                        'processing_time': result.get('processing_time', 0)
                    }
                else:
                    filtered_result = {
                        'success': result['success'],
                        'file_path': result.get('file_path'),
                        'error': result.get('error'),
                        'processing_time': result.get('processing_time', 0)
                    }
                filtered_results.append(filtered_result)
            results = filtered_results
        
        return {
            "success": True,
            "total_files": len(files),
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")
    
    finally:
        # Cleanup all temp files
        for fp in file_paths:
            cleanup_temp_file(fp)

@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get status of async job"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    return JobStatusResponse(
        job_id=job_id,
        status=job['status'],
        result=job.get('result'),
        error=job.get('error'),
        created_at=job['created_at'],
        completed_at=job.get('completed_at')
    )

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete completed job and cleanup resources"""
    
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = jobs[job_id]
    
    if job['status'] in ['pending', 'processing']:
        raise HTTPException(status_code=400, detail="Cannot delete active job")
    
    del jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    
    job_list = []
    for job_id, job_data in jobs.items():
        job_list.append({
            'job_id': job_id,
            'status': job_data['status'],
            'created_at': job_data['created_at'],
            'completed_at': job_data.get('completed_at'),
            'file_name': job_data.get('file_name')
        })
    
    return {
        "total_jobs": len(job_list),
        "jobs": job_list
    }

@app.post("/validate/url")
async def validate_document_from_url(
    url: str,
    options: Dict[str, Any] = {},
    return_detailed: bool = False
):
    """Validate document from URL"""
    
    try:
        import requests
        from urllib.parse import urlparse
        import mimetypes
        
        # Download file
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        # Determine file extension from URL or content type
        parsed_url = urlparse(url)
        file_name = os.path.basename(parsed_url.path)
        
        if not file_name:
            # Try to determine from content-type
            content_type = response.headers.get('content-type', '')
            ext = mimetypes.guess_extension(content_type)
            file_name = f"document{ext or '.pdf'}"
        
        # Save to temp file
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, file_name)
        
        with open(file_path, 'wb') as f:
            f.write(response.content)
        
        # Process document
        result = pipeline.process_document(file_path, options)
        result = convert_numpy_types(result)
        
        # Filter result based on return_detailed flag
        if not return_detailed and result.get('success'):
            filtered_result = {
                'success': result['success'],
                'decision': result.get('decision', {}),
                'processing_time': result.get('processing_time', 0)
            }
        else:
            filtered_result = result
        
        return ValidationResponse(
            success=result.get('success', False),
            result=filtered_result,
            processing_time=result.get('processing_time', 0)
        )
        
    except Exception as e:
        logger.error(f"URL processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to process URL: {str(e)}")
    
    finally:
        try:
            cleanup_temp_file(file_path)
        except:
            pass

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"success": False, "error": "Endpoint not found"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": "Internal server error"}
    )


import numpy as np

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host=config.get('api', {}).get('host', '0.0.0.0'),
        port=config.get('api', {}).get('port', 8000),
        reload=config.get('api', {}).get('debug', False)  # ✅ Use 'reload' instead
    )

