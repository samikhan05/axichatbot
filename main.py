import base64
import time
import subprocess
import json
import soundfile as sf
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from rag_engine import ask, llm

RHUBARB_PATH = r"D:\rhubarb\rhubarb.exe"


def get_mouth_cues(wav_path):
    result = subprocess.run(
        [RHUBARB_PATH, "-f", "json", "--recognizer", "phonetic", wav_path],
        capture_output=True, text=True
    )
    data = json.loads(result.stdout)
    return data["mouthCues"]


def normalize_wav(input_path, output_path):
    data, samplerate = sf.read(input_path)
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


class ChatResponse(BaseModel):
    reply: str


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    answer = ask(request.message)
    return ChatResponse(reply=answer)


@app.post("/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
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

    answer_text = ask(question_text)
    t2 = time.time()
    print(f"RAG+LLM took {t2 - t1:.2f}s")

    print(f"Sending to TTS: {repr(answer_text)}")

    speech_response = llm.audio.speech.create(
        model="canopylabs/orpheus-v1-english",
        voice="daniel",
        input=answer_text,
        response_format="wav",
    )
    audio_data = speech_response.content
    with open("last_response.wav", "wb") as f:
        f.write(audio_data)

    normalize_wav("last_response.wav", "last_response_normalized.wav")
    mouth_cues = get_mouth_cues("last_response_normalized.wav")
    print(f"Mouth cues: {mouth_cues}")

    audio_base64 = base64.b64encode(audio_data).decode("utf-8")
    t3 = time.time()
    print(f"TTS took {t3 - t2:.2f}s")
    print(f"Total: {t3 - t0:.2f}s")

    return {
        "transcript": question_text,
        "reply": answer_text,
        "audio_base64": audio_base64,
        "mouthCues": mouth_cues,
    }