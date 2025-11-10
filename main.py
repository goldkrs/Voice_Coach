# backend/main.py
import os
import io
import re
import json
import tempfile
import requests
import wave
from datetime import datetime, timedelta
import secrets

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, File, UploadFile, Request, Response, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from sqlalchemy import Column, Integer, String, Text, Float, ForeignKey, DateTime, create_engine
from sqlalchemy.orm import sessionmaker, declarative_base, relationship, Session as DBSession

from passlib.context import CryptContext

# speech imports
from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
from pydub import AudioSegment
import numpy as np

# ---------- Config ----------
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")
SECRET_KEY = os.getenv("SECRET_KEY", None)  # not used directly here but reserved
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SESSION_COOKIE_NAME = "session_id"
SESSION_EXPIRE_HOURS = int(os.getenv("SESSION_EXPIRE_HOURS", "24"))

# ---------- FastAPI app ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

# ---------- Whisper + sentiment setup (your existing) ----------
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
sentiment_pipeline = pipeline("sentiment-analysis")

print("OPENAI_API_KEY:", OPENAI_API_KEY)

# ---------- Database (SQLAlchemy) ----------
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(150), unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    transcriptions = relationship("Transcription", back_populates="owner", cascade="all, delete-orphan")

class Transcription(Base):
    __tablename__ = "transcriptions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    transcript = Column(Text)
    filler_count = Column(Integer, default=0)
    silence_duration = Column(Float, default=0.0)
    sentiment = Column(String(32), default="neutral")
    created_at = Column(DateTime, default=datetime.utcnow)
    owner = relationship("User", back_populates="transcriptions")

class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String(128), unique=True, index=True, nullable=False)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    expires_at = Column(DateTime, nullable=False, index=True)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---------- Auth utilities ----------
pwd_context = CryptContext(schemes=["argon2", "bcrypt_sha256"], deprecated="auto")

def _truncate72(pw: str) -> str:
    b = pw.encode("utf-8")
    b72 = b[:72]
    return b72.decode("utf-8", errors="ignore")

def hash_password(password: str) -> str:
    return pwd_context.hash(_truncate72(password))


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)

def create_session(db: DBSession, user_id: int) -> str:
    token = secrets.token_urlsafe(48)
    expires = datetime.utcnow() + timedelta(hours=SESSION_EXPIRE_HOURS)
    s = Session(session_id=token, user_id=user_id, expires_at=expires)
    db.add(s)
    db.commit()
    return token

def get_user_by_session(db: DBSession, session_token: str):
    if not session_token:
        return None
    row = db.query(Session).filter(Session.session_id == session_token).first()
    if not row:
        return None
    if row.expires_at < datetime.utcnow():
        db.delete(row)
        db.commit()
        return None
    return db.query(User).filter(User.id == row.user_id).first()

def delete_session(db: DBSession, session_token: str):
    if not session_token:
        return
    row = db.query(Session).filter(Session.session_id == session_token).first()
    if row:
        db.delete(row)
        db.commit()

# ---------- Pydantic models for auth/transcriptions ----------
class UserCreate(BaseModel):
    username: str
    password: str

class UserOut(BaseModel):
    id: int
    username: str

class TranscriptionOut(BaseModel):
    id: int
    transcript: str
    filler_count: int
    silence_duration: float
    sentiment: str
    created_at: datetime

# ---------- Your existing helper functions (cleaned up) ----------
def detect_fillers_with_llm(transcript: str) -> dict:
    prompt = f"""Analyze the following transcript and identify all filler words used by the speaker. Return a JSON object with each filler word and its count, and a total count. Do not assume any predefined list — infer filler words based on context and usage.

Transcript:
\"\"\"{transcript}\"\"\""""
    try:
        if not OPENAI_API_KEY:
            return {"filler_words": {}, "total_filler_count": 0}
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
            },
            timeout=20,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        print("LLM filler detection error:", e)
        return {"filler_words": {}, "total_filler_count": 0}

def split_sentences(text: str):
    if not text or text.strip() == "":
        return []
    text = re.sub(r'\s+', ' ', text.strip())
    parts = re.split(r'(?<=[.?!])\s+', text)
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
        # if top two are close, treat as neutral
        top = sorted(vals, reverse=True)
        if top[0] - top[1] < 0.15:
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
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
    audio = np.frombuffer(raw_data, dtype=dtype)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)
    audio = audio.astype(np.float32)
    max_abs = np.max(np.abs(audio)) if audio.size else 1.0
    audio /= (max_abs + 1e-8)
    frame_size = int(sample_rate * frame_duration)
    if frame_size <= 0:
        return 0.0
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
    if duration_ms <= 0:
        return segments
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
            task="transcribe",
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
    # tail segment handling
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
                task="transcribe",
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

# ---------- Auth endpoints ----------
def get_current_user_from_cookie(request: Request, db: DBSession = Depends(get_db)):
    token = request.cookies.get(SESSION_COOKIE_NAME)
    user = get_user_by_session(db, token)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user

@app.post("/register", response_model=UserOut)
def register(user_in: UserCreate, response: Response, db: DBSession = Depends(get_db)):
    username = user_in.username.strip()
    if len(username) < 3 or len(user_in.password) < 6:
        raise HTTPException(status_code=400, detail="Invalid username or password length")
    existing = db.query(User).filter(User.username == username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already exists")
    user = User(username=username, hashed_password=hash_password(user_in.password))
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_session(db, user.id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=SESSION_EXPIRE_HOURS * 3600
    )
    return UserOut(id=user.id, username=user.username)

@app.post("/login", response_model=UserOut)
def login(form_data: OAuth2PasswordRequestForm = Depends(), response: Response = None, db: DBSession = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    token = create_session(db, user.id)
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=False,
        samesite="lax",
        max_age=SESSION_EXPIRE_HOURS * 3600
    )
    return UserOut(id=user.id, username=user.username)

@app.post("/logout")
def logout(request: Request, response: Response, db: DBSession = Depends(get_db)):
    cookie = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie:
        delete_session(db, cookie)
    response.delete_cookie(SESSION_COOKIE_NAME)
    return {"ok": True}

@app.get("/me", response_model=UserOut)
def whoami(current_user: User = Depends(get_current_user_from_cookie)):
    return UserOut(id=current_user.id, username=current_user.username)

@app.get("/me/transcriptions", response_model=list[TranscriptionOut])
def list_my_transcriptions(current_user: User = Depends(get_current_user_from_cookie), db: DBSession = Depends(get_db)):
    rows = db.query(Transcription).filter(Transcription.user_id == current_user.id).order_by(Transcription.id.desc()).all()
    return [TranscriptionOut(
        id=r.id,
        transcript=r.transcript,
        filler_count=r.filler_count or 0,
        silence_duration=r.silence_duration or 0.0,
        sentiment=r.sentiment or "neutral",
        created_at=r.created_at
    ) for r in rows]

# ---------- Transcription endpoint (your existing /transcribe, extended to save to DB) ----------
@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), request: Request = None, db: DBSession = Depends(get_db)):
    try:
        # read and prepare audio
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(io.BytesIO(audio_bytes))
        audio = audio.set_channels(1).set_frame_rate(16000)

        # global transcription (your existing)
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

        # compute silence duration using temp wav file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            audio.export(temp_wav.name, format="wav")
            silence_duration = compute_silence_duration(temp_wav.name)

        # filler detection via LLM (if key present)
        filler_result = detect_fillers_with_llm(transcript)
        filler_count = filler_result.get("total_filler_count", 0)

        # segments: 5s windows with 1s overlap
        segments = segment_audio_and_transcribe(audio, window_ms=5000, overlap_ms=1000)

        sentiment_overall = analyze_sentiment(transcript)

        # attempt to save to DB if user authenticated via cookie
        user = None
        try:
            token = request.cookies.get(SESSION_COOKIE_NAME) if request else None
            user = get_user_by_session(db, token) if token else None
        except Exception:
            user = None

        saved_id = None
        if user:
            t = Transcription(
                user_id=user.id,
                transcript=transcript,
                filler_count=filler_count,
                silence_duration=silence_duration,
                sentiment=sentiment_overall
            )
            db.add(t)
            db.commit()
            db.refresh(t)
            saved_id = t.id

        return {
            "transcript": transcript,
            "filler_count": filler_count,
            "silence_duration": silence_duration,
            "sentiment": sentiment_overall,
            "sentiment_segments": segments,
            "saved_id": saved_id
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