"""
language.py — Lightweight language / script detection for English, Urdu
(Nastaliq script) and Roman Urdu (Urdu typed with Latin letters).

No external API or model: Urdu script is detected by Unicode range, and
Roman Urdu is separated from English with a lexicon of high-frequency
Roman Urdu function words (hai, kya, ka, ki, mein, nahi ...). Words that
are also common English words ("to", "me", "is", "main", "the") are left
out of the lexicon on purpose so English sentences aren't misclassified.
"""

import re

EN = "en"
UR = "ur"              # Urdu in Arabic/Nastaliq script
ROMAN_UR = "roman_ur"  # Urdu written in Latin letters

URDU_SCRIPT_RE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
LATIN_WORD_RE = re.compile(r"[a-zA-Z']+")

ROMAN_URDU_WORDS = {
    # verbs / copulas
    "hai", "hain", "hay", "hy", "ho", "hoon", "hun", "tha", "thi", "thay",
    "hota", "hoti", "hotay", "hote", "raha", "rahi", "rahe", "rahay",
    "gaya", "gayi", "gaye", "diya", "diye", "karna", "karta", "karti",
    "karte", "karo", "karein", "karen", "kijiye", "kijye", "dein", "dena",
    "milna", "milne", "milta", "milti", "chahiye", "chahye", "chahta",
    "chahti", "batao", "bataen", "bataein", "bataiye", "bataye", "btao",
    "bolo", "boliye", "sakta", "sakti", "sakte", "jana", "jaana", "aana",
    "ana", "lena", "dekh", "dekho", "pata", "samajh",
    # pronouns / question words
    "kya", "kia", "kyaa", "kaun", "kon", "kaunsa", "konsa", "konsi",
    "kahan", "kahaan", "kidhar", "kab", "kaise", "kese", "kaisay", "kitna",
    "kitne", "kitni", "kyun", "kyon", "kiun", "kis", "kisi", "kuch", "koi",
    "mujhe", "muje", "mjhe", "mera", "meri", "mere", "hum", "hamara",
    "hamari", "aap", "ap", "aapka", "apka", "aapki", "apki", "aapke",
    "apke", "tum", "tumhara", "unka", "unki", "unke", "inka", "inki",
    "yeh", "ye", "woh", "wo", "wahan", "yahan", "idhar", "udhar",
    # postpositions / particles
    "ka", "ke", "ki", "ko", "se", "ne", "mein", "mai", "pe", "tak",
    "bhi", "aur", "ya", "lekin", "magar", "sirf", "bas", "wala", "wali",
    "wale", "walay", "nahi", "nahin", "nai", "mat", "haan", "han", "ji",
    "jee", "na",
    # common nouns / courtesy
    "acha", "accha", "achha", "theek", "thik", "shukriya", "meherbani",
    "salam", "assalam", "assalamualaikum", "alaikum", "walaikum", "janab",
    "sahab", "sahib", "baji", "bhai", "daftar", "dafter", "mulaqat",
    "waqt", "kaam", "kam", "log", "logon", "sab", "sabhi", "abhi",
    "kal", "aaj", "baad", "pehle", "zyada", "kitnay",
}


def has_urdu_script(text: str) -> bool:
    return bool(URDU_SCRIPT_RE.search(text or ""))


def has_devanagari(text: str) -> bool:
    return bool(DEVANAGARI_RE.search(text or ""))


def roman_urdu_score(text: str) -> tuple[int, int]:
    """Returns (lexicon_hits, total_latin_words)."""
    words = [w.lower().strip("'") for w in LATIN_WORD_RE.findall(text or "")]
    words = [w for w in words if w]
    hits = sum(1 for w in words if w in ROMAN_URDU_WORDS)
    return hits, len(words)


def detect_language(text: str, fallback: str | None = None) -> str:
    """Classify text as 'en', 'ur' or 'roman_ur'.

    `fallback` is used for very short / ambiguous messages ("ok", "thanks",
    a bare name) so the conversation keeps the visitor's previous language
    instead of flipping back to English.
    """
    text = (text or "").strip()
    if not text:
        return fallback or EN

    if has_urdu_script(text):
        return UR

    hits, total = roman_urdu_score(text)
    if total == 0:
        return fallback or EN

    ratio = hits / total
    if hits >= 2 and ratio >= 0.2:
        return ROMAN_UR
    if hits >= 1 and total <= 3 and ratio >= 0.5:
        # short messages like "salam", "shukriya", "acha theek"
        return ROMAN_UR

    if hits == 0 and total <= 2 and fallback:
        # "ok", "thanks", "Ahmed?" — too short to judge, keep previous language
        return fallback
    return EN


LANGUAGE_INSTRUCTIONS = {
    EN: "The visitor is writing in English. Reply in English.",
    UR: (
        "The visitor is writing in Urdu (Urdu script). Reply in natural, "
        "polite Urdu using Urdu script. Keep names, email addresses, phone "
        "numbers and technical product names (e.g. FastAPI, React) in "
        "English letters. Use everyday Pakistani Urdu vocabulary, not "
        "Hindi words."
    ),
    ROMAN_UR: (
        "The visitor is writing in Roman Urdu (Urdu typed in English "
        "letters, e.g. 'Ahmed sahab ka email kya hai?'). Reply in Roman "
        "Urdu in the same polite, conversational style — not in English "
        "and not in Urdu script. Keep names, emails, phone numbers and "
        "technical terms exactly as they are. "
        + "Use everyday Pakistani Urdu, never Hindi vocabulary: say waqt "
        "(not samay), shukriya (not dhanyavaad), meherbani (not kripya), "
        "sawal (not prashn), and don't add Hindi phrases like 'aapka din "
        "shubh ho'."
    ),
}
