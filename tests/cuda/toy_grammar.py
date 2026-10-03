"""A toy grammar for the CUDA engines' tests: token t spells t in base 22 over a JSON-like alphabet, 0 ends a reply."""

ALPHA = "abcdefgh{}:,0123456789"
RULE = 'root ::= "{" ([a-h]+ ":" [0-9]+ ",")* [a-h]+ ":" [0-9]+ "}"'


def spell(t: int) -> str:
    out = ""
    while t:
        t, d = divmod(t, len(ALPHA))
        out = ALPHA[d] + out
    return out


def toy(vocab: int):
    """(``Grammars`` over ``vocab`` tokens, the compiled rule); every token's spelling is its own."""

    import xgrammar as xgr

    from tensorfold.engine import grammar

    words = [""] + [spell(t) for t in range(1, vocab)]
    grammars = grammar.Grammars(xgr.TokenizerInfo(words, xgr.VocabType.RAW, vocab_size=vocab, stop_token_ids=[0]))
    return grammars, grammars.compile(grammar.Spec("grammar", RULE))
