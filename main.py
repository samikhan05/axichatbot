import os
import sys
import time
from collections import defaultdict

import requests
from dotenv import load_dotenv

load_dotenv(override=True)

# Urdu text in log lines must not crash the server on a non-UTF-8 console
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from rag_engine import ask_with_meta, llm, MODEL_NAME
from language import detect_language
from speech import transcribe, synthesize_stream, warm_up_tts

conversation_store = defaultdict(list)
MAX_HISTORY = 6

SIMLI_API_KEY = os.getenv("SIMLI_API_KEY")
SIMLI_FACE_ID = os.getenv("SIMLI_FACE_ID")
print(f"Using Simli face: {SIMLI_FACE_ID}")


def _simli_request(method, url, **kwargs):
    """Request to Simli with a 3-attempt retry loop and timeout to handle
    transient network hiccups or rate-limit glitches."""
    last_err = None
    for attempt in range(3):
        try:
            res = requests.request(method, url, timeout=10, **kwargs)
            res.raise_for_status()
            return res
        except requests.exceptions.RequestException as e:
            last_err = e
            print(f"Simli request failed (attempt {attempt + 1}/3): {e}")
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                raise  # bad key / bad request: retrying won't help
    raise last_err


app = FastAPI()

# open Azure TTS connections now, so the first reply is not slowed by setup
warm_up_tts()

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
    language: str = "en"
    sources: list[dict] = []
    tools: list[dict] = []


def _answer(session_id: str, question: str, language: str | None = None) -> dict:
    history = conversation_store[session_id]
    result = ask_with_meta(question, conversation_history=history, language=language)
    history.append({"user": question, "assistant": result["answer"], "language": result["language"]})
    if len(history) > MAX_HISTORY:
        conversation_store[session_id] = history[-MAX_HISTORY:]
    return result


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = _answer(request.session_id, request.message)
    return ChatResponse(
        reply=result["answer"],
        language=result["language"],
        sources=result["sources"],
        tools=result["tools"],
    )


class TTSRequest(BaseModel):
    text: str
    language: str | None = None


@app.post("/tts")
def tts(request: TTSRequest):
    """Stream speech for a reply as raw 16 kHz mono PCM16.

    Chunks are sent while Azure is still synthesising, so the client can
    start feeding the avatar after ~0.5 s instead of waiting for the full
    sentence. Used for both typed and spoken questions."""
    text = request.text.strip()[:2000]
    if not text:
        raise HTTPException(400, "text is empty")
    language = detect_language(text, fallback=request.language)

    t0 = time.time()
    chunks = synthesize_stream(llm, MODEL_NAME, text, language)
    try:
        # produce the first chunk before responding, so setup errors still
        # return a proper error status instead of a broken stream
        first = next(chunks)
    except StopIteration:
        raise HTTPException(502, "TTS produced no audio")
    except Exception as e:
        print(f"TTS failed: {e}")
        raise HTTPException(502, f"TTS failed: {e}")
    print(f"TTS first audio after {time.time() - t0:.2f}s ({language})")

    def body():
        yield first
        try:
            yield from chunks
        except Exception as e:
            print(f"TTS stream interrupted: {e}")

    return StreamingResponse(
        body(),
        media_type="application/octet-stream",
        headers={"X-Audio-Format": "pcm16; rate=16000; channels=1"},
    )


@app.get("/simli-session")
def get_simli_session():
    """Fetches a short-lived Simli session token + ICE servers server-side."""
    if not SIMLI_API_KEY or not SIMLI_FACE_ID:
        raise HTTPException(503, "SIMLI_API_KEY / SIMLI_FACE_ID missing in .env")
    try:
        return _fetch_simli_session()
    except requests.exceptions.HTTPError as e:
        status = e.response.status_code if e.response is not None else 502
        hint = {
            401: " (check SIMLI_API_KEY in .env)",
            403: " (check SIMLI_API_KEY in .env)",
            402: " (Simli credits used up: add credits in the Simli dashboard)",
        }.get(status, "")
        raise HTTPException(502, f"Simli returned {status}{hint}")
    except requests.exceptions.RequestException as e:
        raise HTTPException(502, f"Could not reach Simli: {e.__class__.__name__}")


def _fetch_simli_session():
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
            "model": "arttalk",
        },
    )
    token_data = token_res.json()
    session_token = token_data.get("session_token") or token_data.get("sessionToken") or token_data

    ice_res = _simli_request(
        "GET",
        "https://api.simli.ai/compose/ice",
        headers={"x-simli-api-key": SIMLI_API_KEY},
    )
    return {"sessionToken": session_token, "iceServers": ice_res.json()}


# Plain `def` (not `async def`): STT, the LLM and TTS are all blocking calls,
# so FastAPI runs this in its threadpool instead of freezing the event loop
# for every other visitor while one request is being processed.
@app.post("/voice-chat")
def voice_chat(
    audio: UploadFile = File(...),
    session_id: str = Form(default="default"),
):
    t0 = time.time()
    # kept in memory — no shared temp file, so concurrent requests can't
    # overwrite each other's audio
    audio_bytes = audio.file.read()

    question_text, spoken_language = transcribe(llm, audio_bytes, audio.filename or "audio.webm")
    t1 = time.time()
    print(f"STT took {t1 - t0:.2f}s | language={spoken_language} | text={question_text!r}")

    if not question_text:
        return {
            "transcript": "",
            "reply": "Sorry, I didn't catch that. Could you please say it again?",
            "language": "en",
            "sources": [],
            "tools": [],
        }

    result = _answer(session_id, question_text, language=spoken_language)
    answer_text = result["answer"]
    print(f"RAG+LLM took {time.time() - t1:.2f}s | Total: {time.time() - t0:.2f}s")

    # Speech is streamed separately via /tts, so the text appears right away
    # and the avatar starts talking as soon as the first audio chunk is ready.
    return {
        "transcript": question_text,
        "reply": answer_text,
        "language": result["language"],
        "sources": result["sources"],
        "tools": result["tools"],
    }
