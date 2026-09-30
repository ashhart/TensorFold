"""Token SHA checks shared by the repeated-prefix regressions."""

import hashlib
import json


def same_tokens(actual, expected):
    digest = lambda tokens: hashlib.sha256(json.dumps([int(t) for t in tokens]).encode()).hexdigest()
    assert digest(actual) == digest(expected)
    return digest(actual)
