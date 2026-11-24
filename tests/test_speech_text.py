from speech import make_speakable, speech_only_fixes


def test_slashes_become_spoken_words():
    assert make_speakable("medical/sick leave", "en") == "medical or sick leave"
    assert make_speakable("medical/sick leave", "roman_ur") == "medical ya sick leave"


def test_timezone_and_symbols_removed():
    assert make_speakable("9 AM to 6 PM (PKT).", "en") == "9 AM to 6 PM."
    assert make_speakable("R&D team", "en") == "R and D team"


def test_urls_and_emails_untouched_on_screen():
    text = "Mail info@axitech.com or see https://www.axitech.com/contact_us"
    assert make_speakable(text, "en") == text


def test_acronyms_spaced_and_emails_spelled_for_urdu_voice():
    out = speech_only_fixes("manager یا HR سے، 9 AM, fatima.malik@axitech.com")
    assert "H R" in out
    assert "9 AM" in out
    assert "fatima dot malik at axitech dot com" in out


def test_connectors_between_english_words_are_merged():
    from speech import merge_connectors
    assert merge_connectors("10 دن کی medical یا sick leave") == "10 دن کی medical ya sick leave"
    # a connector next to Urdu words is left alone
    assert merge_connectors("اور maternity 90 دن") == "اور maternity 90 دن"


def test_split_segments_keeps_punctuation_with_previous_segment():
    from speech import split_segments
    segs = split_segments("Annual paid vacation 20 دن ہے, medical ya sick leave 10 دن.")
    assert segs == [
        ("en", "Annual paid vacation"),
        ("ur", "20 دن ہے,"),
        ("en", "medical ya sick leave"),
        ("ur", "10 دن."),
    ]
