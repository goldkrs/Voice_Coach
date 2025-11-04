from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from pydub import AudioSegment
import numpy as np
import io
import tempfile
import requests
import json
import os
import wave

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Whisper setup
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

# Sentiment pipeline (Hugging Face)
sentiment_pipeline = pipeline("sentiment-analysis")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def detect_fillers_with_llm(transcript: str) -> dict:
    prompt = f"""
    Analyze the following transcript and identify all filler words used by the speaker.
    Return a JSON object with each filler word and its count, and a total count. 
    Do not assume any predefined list — infer filler words based on context and usage.

    Transcript:
    \"\"\"{transcript}\"\"\"
    """
    try:
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3
            }
        )
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print("LLM filler detection error:", e)
        return {"filler_words": {}, "total_filler_count": 0}

def analyze_sentiment(text: str) -> str:
    if not text or text.strip() == "":
        return "neutral"
    try:
        res = sentiment_pipeline(text[:1000])  # cap length to avoid huge inputs
        label = res[0]["label"].lower()
        # Normalize label names to simple set
        if label.startswith("pos"):
            return "positive"
        if label.startswith("neg"):
            return "negative"
        return label
    except Exception as e:
        print("Sentiment analysis error:", e)
        return "unknown"

def compute_silence_duration(wav_path, frame_duration=0.025, silence_threshold=0.01):
    with wave.open(wav_path, "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[sample_width]
    audio = np.frombuffer(raw_data, dtype=dtype)

    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    audio = audio.astype(np.float32)
    audio /= np.max(np.abs(audio)) + 1e-8

    frame_size = int(sample_rate * frame_duration)
    num_frames = len(audio) // frame_size

    silence_frames = 0
    for i in range(num_frames):
        frame = audio[i * frame_size : (i + 1) * frame_size]
        energy = np.mean(frame ** 2)
        if energy < silence_threshold:
            silence_frames += 1

    return round(silence_frames * frame_duration, 2)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        print("Audio duration (ms):", len(audio))
        audio = audio.set_channels(1).set_frame_rate(16000)

        if len(audio) == 0:
            raise ValueError("AudioSegment is empty — decoding failed.")

        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        print("Sample count:", len(samples))

        if len(samples) == 0:
            raise ValueError("Decoded audio has zero samples.")

        max_val = np.abs(samples).max()
        if max_val > 0:
            samples = samples / max_val

        inputs = processor(
            samples,
            sampling_rate=16000,
            return_tensors="pt",
            return_attention_mask=True,
            language="en",
            task="transcribe"
        )

        predicted_ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask)
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=False)[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio.export(temp_wav.name, format="wav")
            print("Exported WAV size (bytes):", os.path.getsize(temp_wav.name))
            silence_duration = compute_silence_duration(temp_wav.name)

        filler_result = detect_fillers_with_llm(transcript)
        filler_count = filler_result.get("total_filler_count", 0)

        sentiment = analyze_sentiment(transcript)

        print("Transcript:", transcript)
        print("Filler count:", filler_count)
        print("Silence duration:", silence_duration)
        print("Sentiment:", sentiment)

        return {
            "transcript": transcript,
            "filler_count": filler_count,
            "silence_duration": silence_duration,
            "sentiment": sentiment,
        }

    except Exception as e:
        print("❌ Transcription error:", e)
        return {
            "transcript": "",
            "filler_count": 0,
            "silence_duration": 0,
            "sentiment": "unknown",
            "error": str(e)
        }