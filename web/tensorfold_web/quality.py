"""The quality suite: prompts and strict scoring, next to the Bench's exactness check.

Exactness says the drafts change nothing; this says whether the answers are right, so two quantizations, two
engines or two configurations can be told apart on quality as well as speed. Four categories:

- Arithmetic: 100 GSM8K problems; the final answer must be the right number.
- Knowledge: 100 MMLU questions; the reply must name one letter, the right one.
- Instructions: 20 checks done in code (length, JSON, case, lists, words to use or avoid).
- Long context: a fact stated early and corrected later in a long text; only the corrected value passes.

Scoring is strict on purpose: the last value counts, an answer that offers two fails, a reasoning block left open
leaves no answer. Each item passes or fails, so two runs compare item by item with McNemar's test.
"""

from __future__ import annotations

import hashlib
import json
import re
from importlib import resources

# a letter stated as the answer: "Answer: C", "ANSWER: C", "the correct choice is C", "option: (C)", "actually C";
# the words in any case, the letter a capital (so "the answer is a bit long" names no letter)
_IS = r"\s+(?i:is|would be|should be|must be)"
_STATED = re.compile(rf"(?:\b(?i:answer)\b(?:{_IS})?\s*:?"                    # "answer C" too
                     rf"|\b(?i:choice|option)\b(?:{_IS}\s*:?|\s*:)"          # "option C is wrong" names none
                     r"|\b(?i:actually)\b[^.]{0,30}?)\s*\(?([ABCD])\b")
GSM_TOKENS, MMLU_TOKENS, IF_TOKENS, LONG_TOKENS = 512, 32, 400, 48
_NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*\d[\d,]*(?:\.\d+)?)?")   # 1/18 is one number, a fraction
# how replies are read: part of the suite's digest, so runs scored by different rules are not compared
SCORING = ("2026-10-13: fractions read as their value; a letter contradicted later fails, by \"answer\", \"choice\" or "
           "\"option\" in any case; a stated letter is not a hedge; GSM8K reads the line after Answer:, a typographic "
           "minus as a minus")


# the wording `items` wraps around each question; changed, it is bumped here so earlier runs are not compared
PROMPTS = "2026-10-11"


def load() -> dict:
    data = json.loads(resources.files("tensorfold_web.data").joinpath("quality.json").read_text(encoding="utf-8"))
    canon = json.dumps(data["categories"], sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    # the reply lengths and the prompts' wording count too: changing either makes another suite
    asked = f"{GSM_TOKENS},{MMLU_TOKENS},{IF_TOKENS},{LONG_TOKENS};{PROMPTS}"
    data["digest"] = hashlib.sha256((canon + SCORING + asked).encode("utf-8")).hexdigest()[:12]
    return data


def answer_text(text: str) -> str:
    """The reply without its reasoning: after the last closed `</think>`, nothing when a `<think>` was left open."""

    if "</think>" in text:
        return text.rsplit("</think>", 1)[1].strip()
    if "<think>" in text:
        return text.split("<think>", 1)[0].strip()
    return text.strip()


def _plain(s: str) -> str:
    """"1,5" is a decimal comma, "1,500" a thousands separator: one digit after the comma says which."""

    s = re.sub(r"(\d),(\d)(?!\d)", r"\1.\2", s)
    return s.replace(",", "")


def _num(s: str) -> float | None:
    try:
        if "/" in s:
            top, bottom = (float(_plain(x)) for x in s.split("/", 1))
            return top / bottom if bottom else None
        return float(_plain(s))
    except ValueError:
        return None


def last_number(text: str) -> float | None:
    found = _NUMBER.findall(text)
    return _num(found[-1]) if found else None


# -- prompts ------------------------------------------------------------------------------------------------------------

def _long_prompt(item: dict, filler: str) -> str:
    """A text of about ``size`` tokens: the fact early, the correction later, the question at the end."""

    per = max(1, len(filler) // 4)
    blocks = max(3, (item["size"] - 200) // per)
    parts = [filler] * blocks
    parts.insert(max(1, blocks // 10), item["fact"])
    parts.insert(max(2, (blocks * 7) // 10), item["correction"])
    return ("Read these notes about the town, then answer the question after them.\n\n" + "\n\n".join(parts)
            + "\n\n" + item["question"])


def items(data: dict, filler: str, only: list[str] | None = None) -> list[dict]:
    """Every item as the Bench sends it: category, id, messages, reply length."""

    out = []
    for cat, block in data["categories"].items():
        if only and cat not in only:
            continue
        for it in block["items"]:
            if cat == "gsm8k":
                text = (f"{it['question']}\n\nSolve it. Work it out briefly, then end with a last line of the form "
                        "'Answer: <number>'.")
                tokens = GSM_TOKENS
            elif cat == "mmlu":
                choices = "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(it["choices"]))
                text = f"{it['question']}\n\n{choices}\n\nReply with the letter of the correct answer only."
                tokens = MMLU_TOKENS
            elif cat == "instructions":
                text, tokens = it["prompt"], IF_TOKENS
            else:
                text, tokens = _long_prompt(it, filler), LONG_TOKENS
            out.append({"category": cat, "id": it["id"], "item": it, "max_tokens": tokens,
                        "messages": [{"role": "user", "content": text}]})
    return out


# -- scoring --------------------------------------------------------------------------------------------------------------

def score(category: str, item: dict, reply: str) -> tuple[bool, str]:
    """Whether the reply passes, and what was read from it."""

    text = answer_text(reply).replace("\u2212", "-")      # a typographic minus is a minus
    if not text:
        return False, "no answer"
    if category == "gsm8k":
        # the line that follows the last "Answer:", not the rest of the reply (a closing sentence may hold a number)
        tail = ((text.rsplit("Answer:", 1)[1].strip().splitlines() or [""])[0] if "Answer:" in text
                else text.splitlines()[-1])
        found = [_num(x) for x in _NUMBER.findall(tail)]
        found = [x for x in found if x is not None]
        if not found:
            return False, "no number"
        if len(set(found)) > 1 and re.search(r"\bor\b", tail):
            return False, f"hedged between {', '.join(f'{x:g}' for x in found)}"
        got = found[-1]
        return got == _num(item["answer"]), f"{got:g}"
    if category == "mmlu":
        # a letter alone, or a letter then a mark: "A good case can be made, but B." starts with the article, not an
        # answer; "C is correct" states one
        m = (re.match(r"^\(?([ABCD])\)?(?:[.):,]|\s*$)", text)
             or re.match(r"^\(?([ABCD])\)?\s+is\s+(?:the\s+)?(?:correct|right|answer)\b", text, re.IGNORECASE)
             or re.search(_STATED, text))
        # two letters offered as alternatives fail, unless one was stated as the answer ("C, because A and B are wrong")
        if not m and re.search(r"\b[ABCD]\b\s*(?:or|/|and)\s*\(?[ABCD]\b", text):
            return False, "hedged between letters"
        if m:
            got = m.group(1)
            # "B. Actually the answer is C.": a letter stated later as the answer, or after "actually", overrides
            later = set(re.findall(_STATED, text[m.end():]))
            if later - {got}:
                return False, f"contradicts itself ({got}, then {', '.join(sorted(later - {got}))})"
        else:
            letters = set(re.findall(r"\b([ABCD])\b", text))
            if len(letters) != 1:
                return False, "no single letter"
            got = letters.pop()
        return got == item["answer"], got
    if category == "long_context":
        got = last_number(text)
        return got is not None and got == _num(item["answer"]), "no number" if got is None else f"{got:g}"
    return check_instructions(item["checks"], text)


def _json_of(text: str) -> object:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    return json.loads(t)


def check_instructions(checks: dict, text: str) -> tuple[bool, str]:
    """Each check in code; the first that fails is the reason."""

    words = re.findall(r"[\w'’-]+", text)
    for name, want in checks.items():
        ok = True
        if name == "words_max":
            ok = len(words) <= want
        elif name == "words_min":
            ok = len(words) >= want
        elif name in ("json_keys", "json_list_len"):
            try:
                obj = _json_of(text)
            except ValueError:
                return False, "not JSON"
            if name == "json_keys":
                ok = isinstance(obj, dict) and all(k in obj for k in want)
            else:
                ok = isinstance(obj, dict) and isinstance(obj.get(want["key"]), list) and len(obj[want["key"]]) == want["n"]
        elif name == "excludes_chars":
            ok = not any(c in text for c in want)
        elif name == "lowercase":
            ok = text == text.lower()
        elif name == "uppercase":
            ok = text == text.upper()
        elif name == "ends_with":
            ok = text.rstrip().endswith(want)
        elif name == "starts_with_any":
            ok = any(text.lstrip().startswith(w) for w in want)
        elif name == "bullets":
            lines = [x for x in text.strip().splitlines() if x.strip()]
            ok = len(lines) == want and all(x.lstrip().startswith("- ") for x in lines)
        elif name == "paragraphs":
            ok = len([p for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]) == want
        elif name == "word_count":
            ok = len(re.findall(rf"\b{re.escape(want['word'])}\b", text, re.IGNORECASE)) >= want["min"]
        elif name == "contains_all":
            ok = all(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in want)
        elif name == "excludes_words":
            ok = not any(re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE) for w in want)
        elif name == "exact":
            ok = text.strip().rstrip(".") == want
        elif name == "sentences_max":
            ok = len([s for s in re.split(r"[.!?]+(?:\s|$)", text.strip()) if s.strip()]) <= want
        elif name == "title_brackets":
            ok = bool(re.match(r"^\s*<<[^<>\n]+>>\s*$", text.strip().splitlines()[0]))
        elif name == "wrapped_quotes":
            t = text.strip()
            ok = len(t) > 1 and t[0] in "\"“" and t[-1] in "\"”"
        if not ok:
            return False, f"failed {name}"
    return True, "all checks"
