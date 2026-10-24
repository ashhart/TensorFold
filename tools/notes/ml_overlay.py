"""Apply ML-1 subject surfaces onto a NoteStore as extra phrase rows (N3).

Keep every original O.phrases entry. Add new tuples of 2 to 8 keys. Override
existing keys when a surface collides. Drop new tuples of 1 key or more than 8.
"""


def apply(ns, owner, ml_phrases):
    """Mutate O.phrases from ML.phrases. Returns counts of new, overridden, dropped, colliding."""
    O = ns.owner(owner)
    orig = dict(O.phrases)
    counts = {"new": 0, "overridden": 0, "dropped": 0, "colliding": 0}
    for keys, ent in ml_phrases.items():
        if keys in orig:
            if orig[keys] != ent:
                counts["colliding"] += 1
                counts["overridden"] += 1
                O.phrases[keys] = ent
            continue
        if 2 <= len(keys) <= 8:
            O.phrases[keys] = ent
            counts["new"] += 1
        else:
            counts["dropped"] += 1
    return counts
