"""Conservative sentence chunking for bounded-context Breeze synthesis."""
from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

# Keep terminal punctuation with its sentence so each independently generated
# segment has a natural prosodic ending.  The fallback word split handles long
# unpunctuated input such as captions or pasted transcripts.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?。！？])\s+|\n+")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_EN_EVENT = re.compile(
    r"\((?:laugh|laughs|laughing|cough|coughs|clears\s+throat|sigh|sighs|"
    r"sniff|sneeze|groan|gasp|hum)\)",
    re.IGNORECASE,
)
_ZH_EVENT = re.compile(r"\[(?:笑|笑声|咳嗽|清嗓子|叹气|叹息|抽泣|哭|喘息|呼气)\]")


def estimate_speech_frames(tokenizer: Any, text: str) -> int:
    """Estimate Breeze codec frames using the checkpoint-calibrated ratios."""
    events = len(_EN_EVENT.findall(text)) + len(_ZH_EVENT.findall(text))
    stripped = _EN_EVENT.sub(" ", _ZH_EVENT.sub(" ", text)).strip()
    if not stripped:
        return int(events * 5.0) + 8
    ids = tokenizer(stripped, add_special_tokens=False)["input_ids"]
    tokens = tokenizer.convert_ids_to_tokens(ids)
    cjk = sum(1 for token in tokens if _CJK.search(token))
    estimate = cjk * 3.5 + (len(tokens) - cjk) * 4.1 + events * 5.0 + 4.0
    return max(16, round(estimate))


def split_text_to_fit(text: str, fits: Callable[[str], bool]) -> list[str]:
    """Greedily merge natural sentences while the supplied budget permits."""
    normalized = " ".join(text.split())
    if not normalized:
        return []
    if fits(normalized):
        return [normalized]

    sentences = [
        part.strip()
        for part in _SENTENCE_BOUNDARY.split(normalized)
        if part.strip()
    ]
    units: list[str] = []
    for sentence in sentences:
        units.extend(_split_unit_to_fit(sentence, fits))

    chunks: list[str] = []
    current = ""
    for unit in units:
        candidate = f"{current} {unit}" if current else unit
        if fits(candidate):
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def _split_unit_to_fit(text: str, fits: Callable[[str], bool]) -> list[str]:
    if fits(text):
        return [text]
    words = text.split()
    # CJK and other unspaced input needs a character-boundary fallback.
    pieces = words if len(words) > 1 else list(text)
    result: list[str] = []
    separator = " " if len(words) > 1 else ""
    start = 0
    while start < len(pieces):
        low, high = start + 1, len(pieces)
        best = start
        while low <= high:
            middle = (low + high) // 2
            candidate = separator.join(pieces[start:middle])
            if fits(candidate):
                best = middle
                low = middle + 1
            else:
                high = middle - 1
        if best == start:
            # The caller performs a final fit validation and can report that
            # even the smallest unit cannot coexist with the reference prompt.
            best = start + 1
        result.append(separator.join(pieces[start:best]))
        start = best
    return result
