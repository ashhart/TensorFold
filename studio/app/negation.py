"""Things a video prompt names to forbid them ("no photographs"): H3 has no negative prompt, so they tend to appear."""

from __future__ import annotations

import re

REASON = ("The video model has no negative prompt here: a thing the prompt names tends to appear, also when the prompt "
          "says there is none of it. Take the words out and say what is there instead (\"plain dark walls\" for \"no photographs\").")
# what the Studio writes itself, about the sound track and the control layers of a motion reference
OWN = ("Nobody speaks.", "No voices, no speech.", "No speech after the spoken lines.", "No speech.", "non_diegetic_music: N/A",
       "its look is not rendered", "is not rendered", "are not rendered", "These colours are labels only, not the look of the target video.",
       "no skeleton lines, no flat-coloured, depth-map or segmentation-map look", "and no strike lands", "No music")
IDIOMS = ("not only", "no doubt", "no matter", "not just", "no longer than", "no more than", "no less than", "no sooner")
WORD = r"[A-Za-z][A-Za-z'’-]*"
FOUND = re.compile(r"(?<![A-Za-z'’-])(no one|nobody|no|not|never|without|none of|neither|nor|cannot|" + WORD + r"n['’]t)"
                   r"(?![A-Za-z'’-])((?:[ \t]+" + WORD + r"){0,4})", re.IGNORECASE)
FILLER = {"a", "an", "the", "any", "of", "to", "be", "is", "are", "at", "in", "on", "it", "its", "their", "his", "her", "and", "or",
          "ever", "one", "other", "more", "new", "appear", "appears", "anywhere", "there", "that", "this", "with", "for", "into", "from"}
SECTIONS = re.compile(r"\n\s*(overall_soundscape|non_diegetic_music)\s*:", re.IGNORECASE)


def picture_part(prompt: str) -> str:
    """The part of a composed prompt that describes the picture: up to the soundscape heading."""

    cut = SECTIONS.search(prompt)
    return prompt[:cut.start()] if cut else prompt


def find(text: str, limit: int = 8) -> list[dict]:
    """The negated things in ``text`` as ``phrase`` and ``thing``, in reading order, each phrase once."""

    for own in OWN:
        text = re.sub(re.escape(own), " ", text, flags=re.IGNORECASE)
    out, seen = [], set()
    for match in FOUND.finditer(text):
        negation, rest = match.group(1), match.group(2).split()
        if any((negation + " " + " ".join(rest)).lower().startswith(idiom) for idiom in IDIOMS):
            continue
        # the thing: the words that follow, up to three that carry meaning
        kept, thing = [], []
        for word in rest:
            kept.append(word)
            if word.lower() not in FILLER:
                thing.append(word)
            if len(thing) == 3:
                break
        if not thing and negation.lower() not in ("nobody", "no one"):
            continue
        while kept and kept[-1].lower() in FILLER:
            kept.pop()
        phrase = " ".join([negation, *kept])
        if phrase.lower() in seen:
            continue
        seen.add(phrase.lower())
        out.append({"phrase": phrase, "thing": " ".join(thing)})
        if len(out) == limit:
            break
    return out


def check(text: str) -> dict:
    """What the page shows: ``negations`` (see ``find``), and ``reason`` when there are any."""

    found = find(picture_part(text or ""))
    return {"negations": found, "reason": REASON if found else ""}


def check_fields(fields: dict) -> dict:
    """The same over the wizard's own boxes, saying which box each phrase is in."""

    found = []
    for key, label in (("scene", "Scene"), ("camera", "Camera"), ("action", "Action")):
        found += [{**one, "field": label} for one in find(str(fields.get(key) or ""))]
    return {"negations": found[:12], "reason": REASON if found else ""}
