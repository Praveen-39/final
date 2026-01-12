from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import base64
from app.models import PredictRequest, PredictResponse
from app.simple_inference import run_inference  # Using simplified version (no PyTorch)

app = FastAPI(title="SER & Sarcasm Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/v1/health")
async def health_check():
    return {"status": "healthy", "model_status": "loaded"}

@app.post("/api/v1/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    try:
        # Decode base64 audio
        try:
            audio_bytes = base64.b64decode(req.audio_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid base64 audio data: {str(e)}")
        
        # Run inference
        emotion, em_conf, sarcasm, sarcasm_score, transcript, pitch, energy, centroid = run_inference(audio_bytes, req.sample_rate)
        
        return PredictResponse(
            emotion=emotion,
            emotion_confidence=em_conf,
            sarcasm=sarcasm,
            sarcasm_score=sarcasm_score,
            transcript=transcript,
            pitch=pitch,
            energy=energy,
            spectral_centroid=centroid
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/predict-file", response_model=PredictResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Predict emotion and sarcasm from uploaded audio file
    """
    try:
        # Read file content
        audio_bytes = await file.read()
        
        if len(audio_bytes) == 0:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        print(f"\n{'='*60}")
        print(f"FILE UPLOAD ENDPOINT")
        print(f"Filename: {file.filename}")
        print(f"Content-Type: {file.content_type}")
        print(f"Size: {len(audio_bytes)} bytes ({len(audio_bytes)/1024:.2f} KB)")
        print(f"{'='*60}\n")
        
        # Run inference (sample rate will be auto-detected by librosa)
        emotion, em_conf, sarcasm, sarcasm_score, transcript, pitch, energy, centroid = run_inference(audio_bytes, sample_rate=None)
        
        return PredictResponse(
            emotion=emotion,
            emotion_confidence=em_conf,
            sarcasm=sarcasm,
            sarcasm_score=sarcasm_score,
            transcript=transcript,
            pitch=pitch,
            energy=energy,
            spectral_centroid=centroid
        )
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"ERROR IN FILE UPLOAD ENDPOINT")
        print(f"Error: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print("Traceback:")
        traceback.print_exc()
        print(f"{'='*60}\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
