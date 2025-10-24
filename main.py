from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from pydub import AudioSegment
import numpy as np
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Whisper-small
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read uploaded file into memory
        audio_bytes = await file.read()

        # Decode using FFmpeg via pydub
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))  # auto-detects format
        audio = audio.set_channels(1).set_frame_rate(16000)

        # Convert to float32 numpy array
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        samples = samples / np.abs(samples).max()  # normalize

        # Transcribe with Whisper
        inputs = processor(
            samples,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            language="en",
            task="transcribe"
        )
        predicted_ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask)
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        return {"transcript": transcript}

    except Exception as e:
        print("❌ Transcription error:", e)
        return {"error": str(e)}