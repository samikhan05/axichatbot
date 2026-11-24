import pytest

from language import detect_language, EN, UR, ROMAN_UR


@pytest.mark.parametrize("text, expected", [
    ("What is the email of Ahmed Raza?", EN),
    ("Who is the head of the finance department?", EN),
    ("Where is the main office?", EN),
    ("Is the HR team in the office today?", EN),
    ("Ahmed Raza ka email kya hai?", ROMAN_UR),
    ("office kahan hai", ROMAN_UR),
    ("mujhe HR se milna hai", ROMAN_UR),
    ("finance department ka head kaun hai?", ROMAN_UR),
    ("Assalam o alaikum", ROMAN_UR),
    ("shukriya", ROMAN_UR),
    ("ہیڈ آفس کہاں ہے؟", UR),
    ("Ahmed Raza کا ای میل کیا ہے", UR),
])
def test_detect_language(text, expected):
    assert detect_language(text) == expected


def test_short_message_keeps_previous_language():
    assert detect_language("ok", fallback=ROMAN_UR) == ROMAN_UR
    assert detect_language("thanks", fallback=UR) == UR
    assert detect_language("ok") == EN
