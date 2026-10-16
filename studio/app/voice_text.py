"""Voice take text: speaker tags, directions, parts, voice sheets and word scores."""

from __future__ import annotations

import difflib
import re


SPEAKER = re.compile(r"<\|speaker:(\d+)\|>")
DIRECTION = re.compile(r"\[[^\]]*\]")
SAMPLE_LINE = "I have been waiting for you. Sit down, and tell me everything, from the very beginning."


def pieces(text: str, limit: int) -> list[str]:
    """Text cut at sentence ends into parts of about ``limit`` words, each opening on its speaker (CUDA: 4096 rows)."""

    if len(words(spoken(text))) <= limit:
        return [text]
    sentences, speaker = [], None
    for part in re.split(r"(<\|speaker:\d+\|>)", text):
        if SPEAKER.fullmatch(part):
            speaker = part
            continue
        for sentence in re.split(r"(?<=[.!?…])\s+", part.strip()):
            if sentence.strip():
                sentences.append((speaker, sentence.strip()))
    out, current, count, said = [], [], 0, None
    for who, sentence in sentences:
        size = len(words(spoken(sentence)))
        if current and count + size > limit:
            out.append(" ".join(current))
            current, count, said = [], 0, None
        if who and who != said:
            current.append(who)
            said = who
        current.append(sentence)
        count += size
    if current:
        out.append(" ".join(current))
    return [one for one in out if spoken(one)]


def dialogue(text: str) -> str:
    """With two or more speaker tags each turn goes on a line of its own: on one line Drama 3 drops later turns."""

    if len(SPEAKER.findall(text)) < 2:
        return text
    first = SPEAKER.search(text)
    head, rest = text[: first.end()], text[first.end():]
    return head + re.sub(r"[ \t]*\n?[ \t]*(<\|speaker:\d+\|>)", r"\n\1", rest)


def one_speaker(text: str) -> bool:
    return set(SPEAKER.findall(text)) <= {"0"}



def spoken(text: str) -> str:
    """The words a take should say: speaker tags and [directions] taken out."""

    return " ".join(DIRECTION.sub(" ", SPEAKER.sub(" ", text)).split())


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))


def expected_seconds(text: str) -> float:
    """Roughly how long the take will be, in seconds: about 2.2 words a second."""

    return max(1.5, len(words(spoken(text))) / 2.2)


def sheet_text(sheet: dict) -> str:
    """A Slate voice sheet as a Drama 3 line: its description as a [direction] (not read out), then its sample line."""

    who = " ".join(str(sheet.get("ageGender") or "").split()).rstrip(".")
    sound = ", ".join(" ".join(str(sheet.get(key) or "").split()).rstrip(".") for key in
                      ("accent", "timbre", "pitch", "pacing", "energy", "texture") if str(sheet.get(key) or "").strip())
    direction = ", ".join(bit for bit in (f"the voice of {who}" if who else "", sound) if bit)
    direction = direction.replace("[", "(").replace("]", ")")[:400]
    line = " ".join(str(sheet.get("sampleLine") or "").split()) or SAMPLE_LINE
    return (f"[{direction}] " if direction else "") + line


def match(text: str, heard: str) -> dict:
    wanted, said = words(spoken(text)), words(heard)
    found = sum(1 for one in wanted if one in set(said))
    ratio = difflib.SequenceMatcher(None, wanted, said).ratio() if wanted and said else 0.0
    return {"heard": heard, "score": round(100 * ratio), "found": found, "words": len(wanted)}



def checked_by_engine(measured: dict) -> list[dict] | None:
    """The MLX engine's end-of-text check, part by part (``verify`` in its stats), as kept on the take."""

    parts = measured.get("verify")
    if not isinstance(parts, list) or not parts:
        return None
    keep = ("words", "matched", "missing", "tail_words", "action", "attempts", "cut_s", "late_s", "heard", "asr_s", "reason")
    return [{k: one.get(k) for k in keep if k in one} for one in parts if isinstance(one, dict)] or None


def engine_score(parts: list[dict]) -> dict:
    """A take's word score from the engine's check: words heard of words written, over all its parts."""

    total = sum(int(one.get("words") or 0) for one in parts)
    found = sum(int(one.get("matched") or 0) for one in parts)
    return {"heard": " ".join(str(one.get("heard") or "").strip() for one in parts).strip(), "score": round(100 * found / total) if total else None,
            "found": found, "words": total, "missing": [w for one in parts for w in (one.get("missing") or [])], "by": "engine"}
