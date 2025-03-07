# app.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import os
import shutil
from typing import Optional
import uuid
from model import process_uploaded_image  # Import the function from the file where it's defined

# Create FastAPI app
app = FastAPI(title="Plant Disease Diagnosis API")

# Create uploads directory if it doesn't exist
UPLOAD_DIR = "image"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/api/diagnose-plant-disease/")
async def diagnose_plant_disease(
    file: UploadFile = File(...),
    symptoms: Optional[str] = Form(None)
):
    """
    Upload an image of a plant and get a diagnosis of potential diseases.
    
    - **file**: The image file to upload
    - **symptoms**: Optional description of symptoms observed
    """
    # Validate file is an image
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400, 
            detail="Uploaded file must be an image"
        )
    
    try:
        # Generate a unique filename to prevent collisions
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save the uploaded file
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the image for disease detection
        result = await process_uploaded_image(file_path, symptoms)
        
        # Clean up the file after processing
        if os.path.exists(file_path):
            os.remove(file_path)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        # Clean up on error
        if 'file_path' in locals() and os.path.exists(file_path):
            os.remove(file_path)
            
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )

@app.get("/")
async def root():
    """Health check endpoint"""
    return {"status": "online", "message": "Plant Disease Diagnosis API is running"}