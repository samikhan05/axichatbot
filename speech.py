"""
speech.py — Speech-to-text and text-to-speech for English, Urdu and Roman Urdu.

STT: Groq Whisper large-v3 with automatic language detection. Whisper
     often writes *spoken Urdu* in Hindi (Devanagari) script, because the
     two languages sound almost identical. When that happens we transcribe
     again with language="ur" so the text comes back in Urdu script.

TTS: Azure Neural TTS, streamed. Audio is sent to the client chunk by
     chunk while Azure is still synthesising, so the avatar starts talking
     ~0.5 s after the request instead of waiting for the whole sentence.
     - The voice is chosen from the reply's language.
     - Azure's Urdu voice reads Urdu *script*. Roman Urdu replies (and any
       English words inside Urdu replies) are first written fully in Urdu
       script by a small, fast LLM, so everything is pronounced with a
       natural Pakistani accent instead of being read as English.
     - Text is normalised for speech first: slashes, brackets, "PKT",
       "&" etc. would otherwise be read out literally.
"""

import os
import queue
import re
import threading
import time
from xml.sax.saxutils import escape as xml_escape

from language import UR, ROMAN_UR, has_devanagari, detect_language

AZURE_SPEECH_KEY = os.getenv("AZURE_SPEECH_KEY")
AZURE_SPEECH_REGION = os.getenv("AZURE_SPEECH_REGION", "eastus")
# Urdu replies use two voices: a native Pakistani voice for the Urdu words,
# and a South Asian English voice for the English words inside them. A
# ur-PK voice alone reads English words with Urdu letter rules
# ("maternity" -> "matarnati", "paid" -> "pad"); a multilingual voice alone
# speaks Urdu with a foreign accent. Chosen by ear from voice_samples/.
AZURE_VOICE_UR = os.getenv("AZURE_VOICE_UR", "ur-PK-UzmaNeural")
AZURE_VOICE_UR_ENGLISH = os.getenv("AZURE_VOICE_UR_ENGLISH", "en-IN-NeerjaNeural")
AZURE_VOICE_EN = os.getenv("AZURE_VOICE_EN", "en-US-AvaMultilingualNeural")
HYBRID_POOL_SIZE = int(os.getenv("TTS_POOL_SIZE", "3"))  # warm connections per voice

STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
# defaults to the main chat model: gpt-oss-20b was not faster here and often
# left English words in Latin letters, which the Urdu voice mispronounces
TRANSLIT_MODEL = os.getenv("TRANSLIT_MODEL")


def transcribe(llm, audio_bytes: bytes, filename: str = "audio.webm") -> tuple[str, str]:
    """Returns (transcript, language) where language is 'en', 'ur' or 'roman_ur'."""
    result = llm.audio.transcriptions.create(
        model=STT_MODEL,
        file=(filename, audio_bytes),
        response_format="verbose_json",
    )
    text = (result.text or "").strip()
    whisper_lang = (getattr(result, "language", "") or "").lower()

    if has_devanagari(text) or whisper_lang in ("hindi", "hi"):
        retry = llm.audio.transcriptions.create(
            model=STT_MODEL,
            file=(filename, audio_bytes),
            language="ur",
        )
        text = (retry.text or "").strip()
        return text, UR

    if whisper_lang in ("urdu", "ur"):
        return text, UR
    return text, detect_language(text)


LATIN_WORD_RE = re.compile(r"[A-Za-z]{2,}")


def make_speakable(text: str, language: str) -> str:
    """Rewrite symbols that TTS would read literally."""
    joiner = " or " if language not in (UR, ROMAN_UR) else " ya "
    # protect URLs / emails from the rewrites below
    protected: list[str] = []

    def _protect(m):
        protected.append(m.group(0))
        return f"\x00{len(protected) - 1}\x00"

    text = re.sub(r"(?:https?://|www\.)\S+|\S+@\S+", _protect, text)
    text = re.sub(r"\s*\((?:PKT|PST|GMT|UTC)[^)]*\)", "", text)    # "(PKT)"
    text = re.sub(r"\b(?:PKT|PST)\b", "", text)
    text = re.sub(r"(?<=[^\W\d_])\s*/\s*(?=[^\W\d_])", joiner, text)  # word/word
    text = text.replace("&", " and " if joiner == " or " else " aur ")
    text = re.sub(r"[()\[\]*_#]", " ", text)
    text = re.sub(r"\s+([,.;!?])", r"\1", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: protected[int(m.group(1))], text)
    return re.sub(r"\s{2,}", " ", text).strip()


def to_urdu_script(llm, model: str, text: str) -> str:
    """Put the Urdu words of a Roman Urdu reply into Urdu script for the
    Urdu voice, keeping English words in English letters.

    Measured with a TTS -> Whisper round trip: the ur-PK voice pronounces
    English words correctly when they stay in Latin letters ("paid
    vacation" came back exactly), but garbles them when they are written
    in Urdu script, which has no short-vowel marks ("پیڈ وییکیشن" came back
    as "PED Ocashna")."""
    response = llm.chat.completions.create(
        model=TRANSLIT_MODEL or model,
        messages=[
            {
                "role": "system",
                "content": (
                    "The user's text is Roman Urdu mixed with English. "
                    "Rewrite only the Urdu words in Urdu script. Keep every "
                    "English word exactly as written in English letters: "
                    "e.g. leave, policy, paid vacation, office, manager, "
                    "department, project, Monday, AM, PM, and all names. "
                    "Example: 'Hamari leave policy mein 20 din ki paid "
                    "vacation hai' -> 'ہماری leave policy میں 20 دن کی paid "
                    "vacation ہے'. Keep numbers as digits and emails and URLs "
                    "unchanged. Do not translate, add or drop anything. "
                    "Output only the rewritten text."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0,
        timeout=10.0,
        reasoning_effort="low",
    )
    out = response.choices[0].message.content or ""
    out = re.sub(r"<think>.*?</think>", "", out, flags=re.DOTALL).strip()
    return out or text


ACRONYM_RE = re.compile(r"\b([A-Z]{2,4})\b")
EMAIL_RE = re.compile(r"\b([\w.+-]+)@([\w-]+(?:\.[\w-]+)+)\b")
SPOKEN_ACRONYMS = {"AM", "PM"}  # already read correctly as words


def speech_only_fixes(text: str) -> str:
    """Rewrites that help the Urdu voice but shouldn't appear on screen
    (round-trip tested): "HR" was swallowed, "H R" is read correctly;
    emails were mangled, spelled-out emails are read correctly."""
    text = EMAIL_RE.sub(
        lambda m: f"{m.group(1).replace('.', ' dot ')} at {m.group(2).replace('.', ' dot ')}",
        text,
    )
    return ACRONYM_RE.sub(
        lambda m: m.group(1) if m.group(1) in SPOKEN_ACRONYMS else " ".join(m.group(1)),
        text,
    )


def prepare_speech(llm, model: str, text: str, language: str) -> tuple[str, str]:
    """Returns (text_for_tts, azure_voice)."""
    text = make_speakable(text, language)
    if language in (UR, ROMAN_UR):
        if language == ROMAN_UR:
            # Urdu-script replies need no rewrite: their English words are
            # already in English letters, which the Urdu voice reads best
            t0 = time.time()
            text = to_urdu_script(llm, model, text)
            print(f"Urdu-script rewrite took {time.time() - t0:.2f}s")
        return speech_only_fixes(text), AZURE_VOICE_UR
    return text, AZURE_VOICE_EN


def _azure_configured() -> bool:
    return bool(AZURE_SPEECH_KEY) and not AZURE_SPEECH_KEY.startswith("your_")


# ---------------------------------------------------------------------------
# Warm Azure connections
# A cold Azure request spends ~2-3 s on connection setup before the first
# audio byte; on an already-open connection the first chunk arrives in
# ~0.3 s. So we keep one open synthesizer per voice, open them at startup,
# and re-open them in the background if Azure closes an idle connection.
# ---------------------------------------------------------------------------

_synth_cache: dict = {}            # voice -> (synthesizer, connection, lock)
_synth_cache_lock = threading.Lock()


def _new_synthesizer(voice: str):
    import azure.cognitiveservices.speech as speechsdk

    config = speechsdk.SpeechConfig(subscription=AZURE_SPEECH_KEY, region=AZURE_SPEECH_REGION)
    config.speech_synthesis_voice_name = voice
    config.set_speech_synthesis_output_format(
        speechsdk.SpeechSynthesisOutputFormat.Raw16Khz16BitMonoPcm
    )
    synth = speechsdk.SpeechSynthesizer(speech_config=config, audio_config=None)
    conn = speechsdk.Connection.from_speech_synthesizer(synth)
    conn.open(True)
    return synth, conn


def _cached_synthesizer(voice: str):
    with _synth_cache_lock:
        if voice not in _synth_cache:
            synth, conn = _new_synthesizer(voice)
            _synth_cache[voice] = (synth, conn, threading.Lock())
        return _synth_cache[voice]


def warm_up_tts():
    """Open Azure connections for both voices and keep them open."""
    if not _azure_configured():
        return

    def keep_alive():
        while True:
            for voice in {AZURE_VOICE_UR, AZURE_VOICE_EN}:
                try:
                    _, conn, _ = _cached_synthesizer(voice)
                    conn.open(True)  # no-op when already open, reconnects otherwise
                except Exception as e:
                    print(f"TTS warm-up for {voice} failed: {e}")
            if _hybrid_enabled():
                for voice in (AZURE_VOICE_UR, AZURE_VOICE_UR_ENGLISH):
                    try:
                        for _, conn in _pool_members(voice):
                            conn.open(True)
                    except Exception as e:
                        print(f"TTS pool warm-up for {voice} failed: {e}")
            time.sleep(60)

    threading.Thread(target=keep_alive, daemon=True, name="tts-keepalive").start()


def _ssml_for(text: str, voice: str, speech_lang: str | None) -> str | None:
    """Multilingual voices (e.g. en-US-AvaMultilingualNeural) speak Urdu
    natively when told to via <lang>, and still pronounce the English words
    inside the sentence with real English pronunciation, unlike the ur-PK
    voices, which read "paternity" with Urdu letter rules. Returns None when
    plain text is enough."""
    if not speech_lang or voice.lower().startswith(speech_lang.lower()):
        return None
    # Runs of English words ("paid vacation", "casual") are switched to
    # English pronunciation; without this the voice reads them in Urdu mode
    # ("paid" -> "peed", "casual" -> "cahual").
    body = ENGLISH_RUN_RE.sub(
        lambda m: f"<lang xml:lang='en-US'>{m.group(0)}</lang>", xml_escape(text)
    )
    return (
        "<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' "
        f"xmlns:mstts='https://www.w3.org/2001/mstts' xml:lang='{speech_lang}'>"
        f"<voice name='{voice}'><lang xml:lang='{speech_lang}'>{body}</lang></voice>"
        "</speak>"
    )


# one or more English words (letters, apostrophes, hyphens), space-separated;
# "&amp;"-style escapes never match because ';' is not allowed
ENGLISH_RUN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*(?:\s+[A-Za-z][A-Za-z'\-]*)*")


def azure_tts_stream(text: str, voice: str, speech_lang: str | None = None):
    """Yield raw 16 kHz mono PCM16 chunks while Azure is still synthesising.
    `speech_lang` (e.g. "ur-PK") is passed to multilingual voices."""
    import azure.cognitiveservices.speech as speechsdk

    if not _azure_configured():
        raise RuntimeError("AZURE_SPEECH_KEY is not set in .env")

    synth, _, lock = _cached_synthesizer(voice)
    if not lock.acquire(blocking=False):
        # warm synthesizer busy with another visitor: use a fresh one
        synth, _ = _new_synthesizer(voice)
        lock = None

    chunks: "queue.Queue[bytes | Exception | None]" = queue.Queue()

    def on_canceled(evt):
        detail = speechsdk.CancellationDetails.from_result(evt.result)
        chunks.put(RuntimeError(f"Azure TTS failed: {detail.reason} - {detail.error_details}"))

    for signal in (synth.synthesizing, synth.synthesis_completed, synth.synthesis_canceled):
        signal.disconnect_all()
    synth.synthesizing.connect(lambda evt: chunks.put(evt.result.audio_data))
    synth.synthesis_completed.connect(lambda evt: chunks.put(None))
    synth.synthesis_canceled.connect(on_canceled)

    ssml = _ssml_for(text, voice, speech_lang)
    # runs in the background; chunks arrive through the events above
    future = synth.speak_ssml_async(ssml) if ssml else synth.speak_text_async(text)
    try:
        while True:
            item = chunks.get(timeout=30)
            if item is None:
                break
            if isinstance(item, Exception):
                raise item
            if item:
                yield item
    finally:
        try:
            future.get()  # let this synthesis finish before the next one reuses it
        finally:
            if lock:
                lock.release()


def groq_tts_english(llm, text: str) -> bytes:
    """Fallback English TTS (Groq Orpheus) for when no Azure key is set.
    Returns raw 16 kHz mono PCM16 (no WAV header)."""
    import io
    import numpy as np
    import soundfile as sf
    from scipy.signal import resample_poly

    response = llm.audio.speech.create(
        model="canopylabs/orpheus-v1-english", voice=os.getenv("GROQ_TTS_VOICE", "diana"),
        input=text[:4000], response_format="wav",
    )
    data, sr = sf.read(io.BytesIO(response.content))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        g = np.gcd(int(sr), 16000)
        data = resample_poly(data, 16000 // g, int(sr) // g)
    return (np.clip(data, -1, 1) * 32767).astype("<i2").tobytes()


# ---------------------------------------------------------------------------
# Two-voice Urdu speech without breaks
# Switching voices inside one SSML document leaves ~0.2 s of silence at every
# switch (measured: 17 unwanted breaks in one leave-policy answer). Instead,
# each segment is synthesised separately, in parallel on a pool of warm
# connections, its silent edges are trimmed, and segments are joined with a
# real pause only where there is punctuation (measured: 0 unwanted breaks).
# ---------------------------------------------------------------------------

# small Urdu connector words between two English phrases are spoken by the
# English voice in Roman form, so "medical یا sick leave" is one segment
URDU_CONNECTORS = {
    "یا": "ya", "اور": "aur", "کی": "ki", "کا": "ka", "کے": "ke",
    "سے": "se", "میں": "mein", "کو": "ko", "پر": "par",
}
_CONNECTOR_RE = re.compile(
    r"([A-Za-z][A-Za-z'\-]*)\s+(" + "|".join(URDU_CONNECTORS) + r")\s+(?=[A-Za-z])"
)
_PUNCT_END_RE = re.compile(r"[,.،۔!?؟:;]\s*$")
_PUNCT_START_RE = re.compile(r"^[,.،۔!?؟:;]+\s*")

_pools: dict = {}   # voice -> (queue of (synth, conn), list of members)
_pools_lock = threading.Lock()


def _hybrid_enabled() -> bool:
    return bool(AZURE_VOICE_UR_ENGLISH) and AZURE_VOICE_UR.lower().startswith("ur-")


def _pool_members(voice: str):
    with _pools_lock:
        if voice not in _pools:
            members = [_new_synthesizer(voice) for _ in range(HYBRID_POOL_SIZE)]
            q = queue.Queue()
            for m in members:
                q.put(m)
            _pools[voice] = (q, members)
        return _pools[voice][1]


def _synthesize_full(voice: str, text: str) -> bytes:
    """Whole-segment synthesis on a pooled warm connection -> raw PCM16."""
    import azure.cognitiveservices.speech as speechsdk

    _pool_members(voice)
    q = _pools[voice][0]
    synth, _ = q.get()
    try:
        result = synth.speak_text_async(text).get()
    finally:
        q.put((synth, _))
    if result.reason != speechsdk.ResultReason.SynthesizingAudioCompleted:
        detail = speechsdk.CancellationDetails.from_result(result)
        raise RuntimeError(f"Azure TTS failed: {detail.reason} - {detail.error_details}")
    return result.audio_data


def merge_connectors(text: str) -> str:
    previous = None
    while previous != text:
        previous = text
        text = _CONNECTOR_RE.sub(lambda m: f"{m.group(1)} {URDU_CONNECTORS[m.group(2)]} ", text)
    return text


def split_segments(text: str) -> list[tuple[str, str]]:
    """[(lang, text)] with lang 'ur' or 'en'; punctuation stays with the
    segment before it so pauses land in the right place."""
    raw, pos = [], 0
    for m in ENGLISH_RUN_RE.finditer(text):
        if text[pos:m.start()].strip():
            raw.append(("ur", text[pos:m.start()].strip()))
        raw.append(("en", m.group(0)))
        pos = m.end()
    if text[pos:].strip():
        raw.append(("ur", text[pos:].strip()))

    segments: list[tuple[str, str]] = []
    for lang, seg in raw:
        lead = _PUNCT_START_RE.match(seg)
        if lead and segments:
            segments[-1] = (segments[-1][0], segments[-1][1] + lead.group(0).strip())
            seg = seg[lead.end():]
        if seg.strip():
            segments.append((lang, seg.strip()))
    return segments


def trim_segment(pcm: bytes, text: str, threshold: float = 0.01) -> bytes:
    """Cut the silence Azure adds around every utterance; keep a natural
    pause only after punctuation."""
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2")
    loud = np.nonzero(np.abs(samples) > threshold * 32768)[0]
    if not len(loud):
        return b""
    lead = 160                                                # 10 ms
    tail = 2400 if _PUNCT_END_RE.search(text) else 480        # 150 ms / 30 ms
    start = max(0, int(loud[0]) - lead)
    end = min(len(samples), int(loud[-1]) + tail)
    return samples[start:end].tobytes()


def hybrid_tts_stream(text: str):
    """Urdu voice for Urdu words, English voice for English words, joined
    without breaks. Segments are synthesised in parallel and yielded in
    order, so speech starts as soon as the first segment is ready."""
    from concurrent.futures import ThreadPoolExecutor

    segments = split_segments(merge_connectors(text))
    voices = {"ur": AZURE_VOICE_UR, "en": AZURE_VOICE_UR_ENGLISH}
    with ThreadPoolExecutor(max_workers=HYBRID_POOL_SIZE * 2) as pool:
        futures = [pool.submit(_synthesize_full, voices[lang], seg) for lang, seg in segments]
        for (lang, seg), future in zip(segments, futures):
            audio = trim_segment(future.result(), seg)
            if audio:
                yield audio


TTS_LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "tts_log.jsonl")


def _log_tts(shown: str, spoken: str, voice: str, language: str) -> None:
    """Append what was shown vs. what the voice actually received, so
    pronunciation problems can be traced to the exact TTS input."""
    import json
    from datetime import datetime

    try:
        os.makedirs(os.path.dirname(TTS_LOG), exist_ok=True)
        with open(TTS_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "time": datetime.now().isoformat(timespec="seconds"),
                "language": language, "voice": voice,
                "shown": shown, "spoken": spoken,
            }, ensure_ascii=False) + "\n")
    except OSError as e:
        print(f"Could not write TTS log: {e}")


def synthesize_stream(llm, model: str, text: str, language: str):
    """Yield raw 16 kHz mono PCM16 chunks for `text` in the right voice."""
    speech_text, voice = prepare_speech(llm, model, text, language)
    _log_tts(text, speech_text, voice, language)
    if voice == AZURE_VOICE_EN and not _azure_configured():
        print("No Azure key: using Groq Orpheus for English TTS (Urdu voice needs Azure)")
        yield groq_tts_english(llm, speech_text)
        return
    if language in (UR, ROMAN_UR) and _hybrid_enabled() and ENGLISH_RUN_RE.search(speech_text):
        yield from hybrid_tts_stream(speech_text)
        return
    speech_lang = "ur-PK" if language in (UR, ROMAN_UR) else None
    yield from azure_tts_stream(speech_text, voice, speech_lang)
