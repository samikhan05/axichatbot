import base64
import time
import os
import requests
import soundfile as sf
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import ask, llm
from collections import defaultdict
from dotenv import load_dotenv
load_dotenv(override=True)

conversation_store = defaultdict(list)
MAX_HISTORY = 6

SIMLI_API_KEY = os.getenv("SIMLI_API_KEY")
SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID")
print(f"Using Simli face: {SIMLI_FACE_ID}")


def _simli_request(method, url, **kwargs):
    """
    Helper function to make requests to Simli with a 3-attempt retry loop
    and timeout to handle transient network hiccups or rate-limit glitches.
    """
    last_err = None
    for attempt in range(3):
        try:
            res = requests.request(method, url, timeout=10, **kwargs)
            res.raise_for_status()
            return res
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"Simli request failed (attempt {attempt + 1}/3): {e}")
    raise last_err


def normalize_wav(input_path, output_path, target_samplerate=16000):
    """
    Convert to mono, 16kHz, PCM16 — required by Simli's audioInputFormat="pcm16".
    """
    data, samplerate = sf.read(input_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if samplerate != target_samplerate:
        import numpy as np
        from scipy.signal import resample
        num_samples = int(len(data) * target_samplerate / samplerate)
        data = resample(data, num_samples)
        samplerate = target_samplerate
    sf.write(output_path, data, samplerate, subtype="PCM_16")


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    history = conversation_store[request.session_id]
    answer = ask(request.message, conversation_history=history)
    history.append({"user": request.message, "assistant": answer})
    if len(history) > MAX_HISTORY:
        conversation_store[request.session_id] = history[-MAX_HISTORY:]
    return ChatResponse(reply=answer)


@app.get("/simli-session")
def get_simli_session():
    """
    Fetches a short-lived Simli session token + ICE servers server-side.
    Uses retry wrapper to prevent ConnectionResetError or transient drops.
    """
    token_res = _simli_request(
        "POST",
        "https://api.simli.ai/compose/token",
        headers={"x-simli-api-key": SIMLI_API_KEY, "Content-Type": "application/json"},
        json={
            "faceId": SIMLI_FACE_ID,
            "handleSilence": True,
            "maxSessionLength": 3600,
            "maxIdleTime": 180,
            "audioInputFormat": "pcm16",
            "model": "arttalk",   # try this if not already set
        },
    )
    token_data = token_res.json()
    session_token = token_data.get("session_token") or token_data.get("sessionToken") or token_data

    ice_res = _simli_request(
        "GET",
        "https://api.simli.ai/compose/ice",
        headers={"x-simli-api-key": SIMLI_API_KEY},
    )
    ice_servers = ice_res.json()

    return {"sessionToken": session_token, "iceServers": ice_servers}


@app.post("/voice-chat")
async def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = Form(default="default")
):
    t0 = time.time()
    audio_bytes = await audio.read()
    temp_path = "temp_input.webm"
    with open(temp_path, "wb") as f:
        f.write(audio_bytes)

    with open(temp_path, "rb") as f:
        transcription = llm.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            language="en",
        )
    question_text = transcription.text
    t1 = time.time()
    print(f"STT took {t1 - t0:.2f}s")

    history = conversation_store[session_id]
    answer_text = ask(question_text, conversation_history=history)
    history.append({"user": question_text, "assistant": answer_text})
    if len(history) > MAX_HISTORY:
        conversation_store[session_id] = history[-MAX_HISTORY:]

    t2 = time.time()
    print(f"RAG+LLM took {t2 - t1:.2f}s")

    speech_response = llm.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="diana",
        input=answer_text,
        response_format="wav",
    )
    audio_data = speech_response.content
    with open("last_response.wav", "wb") as f:
        f.write(audio_data)

    # Normalize to 16kHz mono PCM16 — required by Simli's audioInputFormat
    normalize_wav("last_response.wav", "last_response_normalized.wav")
    with open("last_response_normalized.wav", "rb") as f:
        normalized_bytes = f.read()
    audio_base64 = base64.b64encode(normalized_bytes).decode("utf-8")

    t3 = time.time()
    print(f"TTS took {t3 - t2:.2f}s | Total: {t3 - t0:.2f}s")

    return {
        "transcript": question_text,
        "reply": answer_text,
        "audio_base64": audio_base64,
    }