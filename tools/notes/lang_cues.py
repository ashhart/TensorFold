"""German and Finnish past cues for ML-1 (gate +1).

Regexes copied from fig6-live/multilingual/templates_de.py:183-186 and
templates_fi.py:224-229, minus the N4 verbs that also appear as stems in
those templates. English stays in past.zig (past.zig:111). The chat
question gate is English only (question.zig:1).

Removed from DE: wohnte, wohntest, hieß, mochte, mochtest, gehörte.
Removed from FI: asui, kuului, piti, pidin, kutsuttiin.
"""
import re

# Finnish question clitics, same as templates_fi.py:_CLIT.
FI_CLIT = r"(?:ko|kö|kin|kaan|kään|han|hän)?"

DE_PAST_CUE = re.compile(
    r"\b(?:war|waren|warst|wart|hatte|hatten|hattest|früher|vorher|zuvor|damals|ursprünglich|einst|je|jemals"
    r"|schon\s+immer|immer|geändert|verändert|gewechselt|bis|seit|seitdem"
    r"|vor\s+\w+\s+(?:Jahren|Monaten|Wochen|Tagen))\b|\b(?:19|20)\d\d\b",
    re.I,
)

FI_PAST_CUE = re.compile(
    r"\b(?:oli|olivat|olin|olit|olimme|olitte|ollut|olleet|ennen|aiemmin|aikaisemmin|alun\s+perin|alkujaan"
    r"|entinen|entisen|joskus|koskaan|ikinä|aina|muuttunut|muuttuneet|muuttui|muuttuivat|vaihtunut|vaihtui"
    r"|vaihtanut|asti|saakka|lähtien|alkaen|sitten)"
    + FI_CLIT
    + r"\b"
    r"|\b(?:19|20)\d\d\b",
    re.I,
)


def cue(lang: str):
    if lang == "de":
        return DE_PAST_CUE
    if lang == "fi":
        return FI_PAST_CUE
    raise ValueError(f"lang_cues has no cue for {lang!r}; en uses past.zig")
