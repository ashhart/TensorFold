"""Build a byte-level tokenizer from GGUF metadata, without a native engine dependency."""

from __future__ import annotations


def tokenizer(metadata):
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, pre_tokenizers

    if metadata.get("tokenizer.ggml.model") != "gpt2":
        raise ValueError("GGUF tokenizer: only byte-level GPT2 BPE is currently implemented")
    pre = metadata.get("tokenizer.ggml.pre", "gpt2")
    patterns = {
        "gpt2": [r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"],
        "joyai-llm": [
            r"\p{N}{1,3}",
            r"[一-龥぀-ゟ゠-ヿ]+",
            r"""[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~][A-Za-z]+|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+| ?[\p{P}\p{S}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        ],
    }
    if pre not in patterns:
        raise ValueError(f"GGUF tokenizer pre-type {pre!r} is not implemented")
    tokens = metadata["tokenizer.ggml.tokens"]
    vocab = {t: i for i, t in enumerate(tokens)}
    if len(vocab) != len(tokens):
        raise ValueError("GGUF tokenizer has duplicate token pieces")
    merges = [tuple(m.split(" ")) for m in metadata["tokenizer.ggml.merges"]]
    if any(len(m) != 2 for m in merges):
        raise ValueError("GGUF BPE merge must contain exactly two pieces")
    tok = Tokenizer(models.BPE(vocab, merges))
    splits = [pre_tokenizers.Split(Regex(p), behavior="isolated") for p in patterns[pre]]
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        splits + [pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)]
    )
    tok.decoder = decoders.ByteLevel()
    types = metadata.get("tokenizer.ggml.token_type", [1] * len(tokens))
    if len(types) != len(tokens):
        raise ValueError("GGUF token types must match the vocabulary length")
    special = [AddedToken(t, special=True, normalized=False) for t, kind in zip(tokens, types) if kind in (3, 4)]
    tok.add_special_tokens(special)
    return tok
