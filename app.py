from fastapi import FastAPI, UploadFile, HTTPException, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from summarizer import summarizer
import os
import shutil
import logging

app = FastAPI()

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

# Serve static files (like index.html)
app.mount("/static", StaticFiles(directory="."), name="static")

@app.get("/")
async def read_root():
    return FileResponse('index.html')

# Ensure the temp directory exists
if not os.path.exists('temp'):
    os.makedirs('temp')

@app.post("/upload")
async def upload(file: UploadFile):
    # Check if the file is a text file
    if not file.content_type.startswith('text/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Only text files are accepted.")

    temp_file_path = f"temp/{file.filename}"

    try:
        # Log file upload
        logging.info(f"Processing file: {file.filename}")

        # Save the uploaded file to the temporary file path
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Pass the file path to the summarizer function and get the summary
        summary = summarizer(temp_file_path, method="refine")  # Adjust the method as needed

        # Return the summary
        return {"filename": file.filename, "summary": summary}

    except Exception as e:
        # Log the error
        logging.error(f"Error processing file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Error in summarization process")

    finally:
        file.file.close()
        # Optionally, delete the temporary file after processing
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
