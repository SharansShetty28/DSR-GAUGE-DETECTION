from fastapi import FastAPI, UploadFile, File

app = FastAPI(title="Marine Gauge AI")


@app.get("/")
def root():
    return {"message": "Marine Gauge AI API is running"}


@app.post("/extract")
async def extract_gauge(file: UploadFile = File(...)):
    # TODO: save upload, run the pipeline, and return structured JSON.
    return {
        "filename": file.filename,
        "status": "starter_api",
        "message": "Pipeline connection will be added next."
    }
