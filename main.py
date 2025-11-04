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
import re

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

# Sentiment pipeline
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


def split_sentences(text: str):
    if not text or text.strip() == "":
        return []
    text = re.sub(r'\s+', ' ', text.strip())
    parts = re.split(r'(?<=[\.\?\!])\s+', text)
    parts = [p.strip() for p in parts if p.strip()]
    return parts


def analyze_sentiment(text: str) -> str:
    if not text or text.strip() == "":
        return "neutral"
    try:
        sentences = split_sentences(text)
        if not sentences:
            res = sentiment_pipeline(text[:1000])
            return res[0]["label"].lower()
        n = len(sentences)
        weights = []
        scores = []
        for i, s in enumerate(sentences):
            weight = 1.0 + (i / max(1, n - 1))
            weights.append(weight)
            out = sentiment_pipeline(s[:1000])[0]
            label = out["label"].lower()
            if label.startswith("pos"):
                lab = "positive"
                score = out.get("score", 1.0)
            elif label.startswith("neg"):
                lab = "negative"
                score = out.get("score", 1.0)
            else:
                lab = "neutral"
                score = out.get("score", 1.0)
            scores.append((lab, float(score)))
        agg = {"positive": 0.0, "negative": 0.0, "neutral": 0.0}
        for (lab, sc), w in zip(scores, weights):
            agg[lab] += sc * w
        vals = list(agg.values())
        if max(vals) - sorted(vals)[-2] < 0.15:
            return "neutral"
        return max(agg.items(), key=lambda kv: kv[1])[0]
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
        frame = audio[i * frame_size: (i + 1) * frame_size]
        energy = np.mean(frame ** 2)
        if energy < silence_threshold:
            silence_frames += 1

    return round(silence_frames * frame_duration, 2)


def segment_audio_and_transcribe(audio_segment: AudioSegment, window_ms=5000, overlap_ms=1000):
    duration_ms = len(audio_segment)
    step = window_ms - overlap_ms if window_ms > overlap_ms else window_ms
    segments = []
    for start_ms in range(0, max(1, duration_ms - window_ms + 1), step):
        end_ms = start_ms + window_ms
        seg = audio_segment[start_ms:end_ms]
        if len(seg) < 200:
            continue
        samples = np.array(seg.get_array_of_samples()).astype(np.float32)
        if seg.channels > 1:
            samples = samples.reshape((-1, seg.channels)).mean(axis=1)
        max_val = np.abs(samples).max() if samples.size else 0
        if max_val > 0:
            samples = samples / max_val
        inputs = processor(
            samples,
            sampling_rate=seg.frame_rate,
            return_tensors="pt",
            return_attention_mask=True,
            language="en",
            task="transcribe"
        )
        predicted_ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask)
        seg_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
        seg_sentiment = analyze_sentiment(seg_transcript)
        segments.append({
            "start": round(start_ms / 1000.0, 2),
            "end": round(end_ms / 1000.0, 2),
            "transcript": seg_transcript,
            "sentiment": seg_sentiment
        })
    # tail segment
    last_covered = (((duration_ms - window_ms) // step) * step) if duration_ms >= window_ms else 0
    tail_start = last_covered + step if duration_ms > (last_covered + window_ms) else None
    if tail_start and tail_start < duration_ms:
        seg = audio_segment[tail_start:duration_ms]
        if len(seg) >= 200:
            samples = np.array(seg.get_array_of_samples()).astype(np.float32)
            if seg.channels > 1:
                samples = samples.reshape((-1, seg.channels)).mean(axis=1)
            max_val = np.abs(samples).max() if samples.size else 0
            if max_val > 0:
                samples = samples / max_val
            inputs = processor(
                samples,
                sampling_rate=seg.frame_rate,
                return_tensors="pt",
                return_attention_mask=True,
                language="en",
                task="transcribe"
            )
            predicted_ids = model.generate(inputs.input_features, attention_mask=inputs.attention_mask)
            seg_transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            seg_sentiment = analyze_sentiment(seg_transcript)
            segments.append({
                "start": round(tail_start / 1000.0, 2),
                "end": round(duration_ms / 1000.0, 2),
                "transcript": seg_transcript,
                "sentiment": seg_sentiment
            })
    return segments


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)

        # global transcription
        samples = np.array(audio.get_array_of_samples()).astype(np.float32)
        if audio.channels > 1:
            samples = samples.reshape((-1, audio.channels)).mean(axis=1)
        max_val = np.abs(samples).max() if samples.size else 0
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
        transcript = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio.export(temp_wav.name, format="wav")
            silence_duration = compute_silence_duration(temp_wav.name)

        filler_result = detect_fillers_with_llm(transcript)
        filler_count = filler_result.get("total_filler_count", 0)

        # 5s windows, 1s overlap
        segments = segment_audio_and_transcribe(audio, window_ms=5000, overlap_ms=1000)

        sentiment_overall = analyze_sentiment(transcript)

        return {
            "transcript": transcript,
            "filler_count": filler_count,
            "silence_duration": silence_duration,
            "sentiment": sentiment_overall,
            "sentiment_segments": segments
        }

    except Exception as e:
        print("❌ Transcription error:", e)
        return {
            "transcript": "",
            "filler_count": 0,
            "silence_duration": 0,
            "sentiment": "unknown",
            "sentiment_segments": [],
            "error": str(e)
        }