"""Drama 3's end-of-text word check for the CUDA engine: redraw under 85 % heard, 3 missing in a row, or no ending."""
# Copied, not imported, from TensorFold's verify.py (drama3-mlx 92c09a8e, Apache-2.0); keep its rules the engine's

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass

import numpy as np

# -- text normalisation -------------------------------------------------------------------------------------------

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_SCALES = ((10 ** 12, "trillion"), (10 ** 9, "billion"), (10 ** 6, "million"), (1000, "thousand"))
_ORDINAL = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
            "twelve": "twelfth"}


def number_words(n: int) -> list[str]:
    """English words of a non-negative integer (no "and")."""

    if n < 20:
        return [_ONES[n]]
    if n < 100:
        return [_TENS[n // 10]] + ([] if n % 10 == 0 else [_ONES[n % 10]])
    if n < 1000:
        return [_ONES[n // 100], "hundred"] + ([] if n % 100 == 0 else number_words(n % 100))
    for value, name in _SCALES:
        if n >= value:
            return number_words(n // value) + [name] + ([] if n % value == 0 else number_words(n % value))
    return [str(n)]


def _ordinal(words: list[str]) -> list[str]:
    last = words[-1]
    if last in _ORDINAL:
        last = _ORDINAL[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last = last + "th"
    return words[:-1] + [last]


_TAGS = re.compile(r"<\|[^|]*\|>|\[[^\]]*\]")
_NUMBER = re.compile(r"^(\d{1,3}(?:,\d{3})+|\d+)(?:\.(\d+))?(st|nd|rd|th|s)?$")
_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]")


def _token_words(token: str) -> list[str]:
    """Normalised words of one white-space separated token."""

    token = token.lower().replace("’", "'").replace("‘", "'")
    token = token.replace("%", " percent ").replace("&", " and ").replace("+", " plus ")
    out: list[str] = []
    for piece in re.split(r"[\s\-–—/]+", token):
        piece = piece.strip(".,!?;:\"'()[]{}…«»“”„*_~`")
        if not piece:
            continue
        currency = piece.startswith("$")
        piece = piece.lstrip("$")
        m = _NUMBER.match(piece)
        if m and len(m.group(1).replace(",", "")) <= 15:
            words = number_words(int(m.group(1).replace(",", "")))
            if m.group(2):
                words = words + ["point"] + [_ONES[int(d)] for d in m.group(2)]
            if m.group(3) in ("st", "nd", "rd", "th"):
                words = _ordinal(words)
            elif m.group(3) == "s":                  # the 1990s
                words = words[:-1] + [words[-1][:-1] + "ies" if words[-1].endswith("y") else words[-1] + "s"]
            out += words + (["dollars"] if currency else [])
            continue
        if _CJK.search(piece):
            out += [ch for ch in re.findall(r"[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]|"
                                            r"[^\W_]+", piece)]
            continue
        piece = re.sub(r"[^\w']+", " ", piece).strip()
        out += [w.strip("'") for w in piece.split() if w.strip("'")]
    return out


def words_of(text: str) -> list[str]:
    """The spoken words of text: tags and directions dropped, numbers spelled out, lower case, no punctuation."""

    return [w for token in _TAGS.sub(" ", text).split() for w in _token_words(token)]


def ends_with_direction(text: str) -> bool:
    """Whether a [direction] follows the last spoken word (a trailing laugh or sigh is not a runaway)."""

    tail = re.sub(r"<\|[^|]*\|>", " ", text).rstrip()
    m = re.search(r"\[[^\]]*\]\W*$", tail)
    return m is not None and not words_of(tail[m.end():])


# words whose absence at the very end is more likely the recogniser's than the take's
FUNCTION_WORDS = frozenset("a an the to of and or but in on at by for with from up it is be am are was so too oh ah "
                           "uh um hm mm yeah yes no me you we us he him she her they them i my our its".split())


def last_is_content(text: str) -> bool:
    """Whether the last spoken word is one whose absence means a dropped ending: not a function word or a number."""

    spoken = _TAGS.sub(" ", text).split()
    if not spoken or not words_of(spoken[-1]):
        return False
    last = spoken[-1]
    if re.search(r"\d", last):
        return False
    return words_of(last)[-1] not in FUNCTION_WORDS


# -- alignment ----------------------------------------------------------------------------------------------------

MATCH_SIMILARITY = 0.6
MERGE_COST = 0.2                     # a split / merge needs similarity >= 0.8


def similarity(a: str, b: str) -> float:
    return 1.0 if a == b else difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def align(ref: list[str], hyp: list[str]) -> list[tuple[list[int], list[int]]]:
    """Minimum-cost alignment of reference and hypothesis words, as (reference indices, hypothesis indices) steps."""

    n, m = len(ref), len(hyp)
    INF = float("inf")
    cost = np.full((n + 1, m + 1), INF)
    back: dict = {}
    cost[0, 0] = 0.0
    sim_cache: dict = {}

    def sub(a: str, b: str) -> float:
        """0 same word, up to 0.4 near, 1.5 different (under a miss plus an extra, over either alone)."""

        key = (a, b)
        if key not in sim_cache:
            if a == b:
                sim_cache[key] = 0.0
            else:
                matcher = difflib.SequenceMatcher(None, a, b, autojunk=False)
                # the cheap upper bound settles clear mismatches; only near words get the exact ratio
                sim = matcher.ratio() if matcher.quick_ratio() >= MATCH_SIMILARITY else 0.0
                sim_cache[key] = 1.0 - sim if sim >= MATCH_SIMILARITY else 1.5
        return sim_cache[key]

    for i in range(n + 1):
        for j in range(m + 1):
            here = cost[i, j]
            if here == INF:
                continue
            moves = []
            if i < n:
                moves.append((i + 1, j, 1.0, ([i], [])))
            if j < m:
                moves.append((i, j + 1, 1.0, ([], [j])))
            if i < n and j < m:
                moves.append((i + 1, j + 1, sub(ref[i], hyp[j]), ([i], [j])))
            # a split or merge must be near-exact, or a dropped short word hides in it ("the" for "the cat")
            if i < n and j + 1 < m and (c := sub(ref[i], hyp[j] + hyp[j + 1])) <= MERGE_COST:
                moves.append((i + 1, j + 2, c + 0.25, ([i], [j, j + 1])))
            if i + 1 < n and j < m and (c := sub(ref[i] + ref[i + 1], hyp[j])) <= MERGE_COST:
                moves.append((i + 2, j + 1, c + 0.25, ([i, i + 1], [j])))
            for a, b, c, step in moves:
                if here + c < cost[a, b]:
                    cost[a, b] = here + c
                    back[(a, b)] = (i, j, step)
    path = []
    a, b = n, m
    while (a, b) != (0, 0):
        i, j, step = back[(a, b)]
        path.append(step)
        a, b = i, j
    return path[::-1]


# -- the check ------------------------------------------------------------------------------------------------------

@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Check:
    words: int                       # spoken words in the text
    matched: int
    coverage: float
    longest_missing_run: int
    missing: list[str]
    extra_inside: int                # heard words aligned to no text word, between text words
    tail_words: list[str]            # heard after the last aligned text word
    last_word_end_s: float | None
    audio_s: float
    tail_s: float                    # audio after the last aligned word
    action: str                      # "ok", "trim" or "redraw"
    ending_missing: bool = False     # the text's last content word (or its last two words) was not heard
    reason: str = ""
    cut_s: float | None = None       # where a trim cuts the audio
    heard: str = ""
    asr_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.action != "redraw"

    @property
    def score(self) -> tuple:
        return (self.ok, round(self.coverage, 3), -self.longest_missing_run, -len(self.tail_words))

    def summary(self) -> str:
        text = f"{self.matched}/{self.words} words"
        if self.missing:
            text += f" (missing {' '.join(self.missing[:6])}{' ...' if len(self.missing) > 6 else ''})"
        if self.tail_words:
            text += f", {len(self.tail_words)} words after the text ({' '.join(self.tail_words[:6])})"
        elif self.tail_s > 0:
            text += f", {self.tail_s:.1f}s after the last word"
        return f"{text}: {self.action}" + (f" ({self.reason})" if self.reason else "")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["coverage"] = round(self.coverage, 3)
        for k in ("last_word_end_s", "audio_s", "tail_s", "cut_s", "asr_s"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        return d


@dataclass
class Policy:
    min_coverage: float = 0.85       # a take below this redraws
    max_missing_run: int = 2         # more consecutive missing words redraws
    max_tail_s: float = 1.2          # longer sound after the last word is trimmed...
    max_tail_after_direction_s: float = 4.0   # ...unless the text ends with a [direction] (a laugh, a sigh)
    tail_pad_s: float = 0.3          # audio kept after the last word's end when trimming
    fade_s: float = 0.04


def check_words(text: str, heard: list[Word], audio_s: float, policy: Policy = Policy()) -> Check:
    """Align heard words (with times) to text and decide what to do with the take."""

    ref = words_of(text)
    hyp_words: list[Word] = []
    for w in heard:                                  # one heard word may normalise to several ("41." -> forty one)
        parts = words_of(w.text)
        span = (w.end - w.start) / max(1, len(parts))
        hyp_words += [Word(p, float(w.start + k * span), float(w.start + (k + 1) * span))
                      for k, p in enumerate(parts)]
    hyp = [w.text for w in hyp_words]
    path = align(ref, hyp)
    matched_ref: set[int] = set()
    last_ref_hyp = -1
    for ri, hj in path:
        if ri and hj and 1.0 - similarity("".join(ref[i] for i in ri), "".join(hyp[j] for j in hj)) \
                <= 1.0 - MATCH_SIMILARITY:
            matched_ref.update(ri)
            last_ref_hyp = max(last_ref_hyp, max(hj))
    missing_idx = [i for i in range(len(ref)) if i not in matched_ref]
    run = longest = 0
    for i in range(len(ref)):
        run = run + 1 if i not in matched_ref else 0
        longest = max(longest, run)
    tail = hyp[last_ref_hyp + 1:] if last_ref_hyp >= 0 else hyp
    first_aligned = min((min(hj) for ri, hj in path if ri and hj), default=0)
    extra_inside = sum(1 for ri, hj in path if hj and not ri and first_aligned < hj[0] <= last_ref_hyp)
    last_end = hyp_words[last_ref_hyp].end if last_ref_hyp >= 0 else None
    tail_s = audio_s - last_end if last_end is not None else audio_s
    coverage = len(matched_ref) / max(1, len(ref))
    result = Check(words=len(ref), matched=len(matched_ref), coverage=coverage, longest_missing_run=longest,
                   missing=[ref[i] for i in missing_idx], extra_inside=extra_inside, tail_words=tail,
                   last_word_end_s=last_end, audio_s=audio_s, tail_s=max(0.0, tail_s), action="ok",
                   heard=" ".join(w.text.strip() for w in heard))
    if not ref:
        return result
    last_two = bool(len(ref) >= 2) and (len(ref) - 1) in missing_idx and (len(ref) - 2) in missing_idx
    last_content = bool(ref) and (len(ref) - 1) in missing_idx and last_is_content(text)
    ending_missing = last_two or last_content
    result.ending_missing = ending_missing
    if coverage < policy.min_coverage:
        result.action, result.reason = "redraw", f"coverage {coverage:.2f} < {policy.min_coverage}"
    elif longest > policy.max_missing_run:
        result.action, result.reason = "redraw", f"{longest} words in a row missing"
    elif ending_missing:
        result.action, result.reason = "redraw", ("the last word is missing" if not last_two
                                                  else "the ending is missing")
    else:
        allowed = policy.max_tail_after_direction_s if ends_with_direction(text) else policy.max_tail_s
        if last_end is not None and (tail or tail_s > allowed):
            cut = last_end + (allowed if not tail and ends_with_direction(text) else policy.tail_pad_s)
            if tail:                                  # do not keep the start of the next (extra) word
                nxt = hyp_words[last_ref_hyp + 1].start
                cut = min(cut, max(last_end + 0.05, (last_end + nxt) / 2 + 0.05))
            if cut < audio_s - 0.05:
                result.action = "trim"
                result.cut_s = cut
                result.reason = (f"{len(tail)} words after the text" if tail
                                 else f"{tail_s:.1f}s of sound after the last word")
    return result


def heard_words(transcript: str) -> list[Word]:
    """A transcript as heard words (no timestamps: the check then judges words only, never a tail of sound)."""

    return [Word(token, 0.0, 0.0) for token in (transcript or "").split()]


def check_text(text: str, transcript: str) -> dict:
    """The engine's check of text against what Whisper heard, as a dict (action, words, matched, missing, ...)."""

    return check_words(text, heard_words(transcript), 0.0).as_dict()
