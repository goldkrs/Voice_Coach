from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from scipy.io import wavfile
import numpy as np

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Step 2: Use Whisper-medium for better accuracy
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        sr, waveform = wavfile.read(file.file)
        waveform = waveform.astype(np.float32)

        # ✅ Normalize safely
        max_val = np.abs(waveform).max()
        if max_val > 0:
            waveform = waveform / max_val

        # ✅ Step 3: Pad short audio to ~1 second (16000 samples)
        if waveform.shape[0] < 16000:
            padded = np.zeros(16000, dtype=np.float32)
            start = (16000 - waveform.shape[0]) // 2
            padded[start:start + waveform.shape[0]] = waveform
            waveform = padded

        inputs = processor(
            waveform.squeeze(),
            sampling_rate=sr,
            return_tensors="pt",
            return_attention_mask=True,
            language="en",  # Force English transcription
            task="transcribe"
        )
        predicted_ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask)
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return {"transcript": transcript}
    except Exception as e:
        print("❌ Transcription error:", e)
        return {"error": str(e)}