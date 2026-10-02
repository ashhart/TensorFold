"""Add caller-owned sampling and prefill checkpoints to a verified build copy.

The cached upstream source stays untouched. All four continuous target sampling
sites already call this one function, including speculative verification rows.
The checkpoint callback runs after target prefill and drafter KV injection;
attached drafters maintain that prefix even while block drafting is disabled.
"""

from pathlib import Path

ANCHOR = """        uint64_t    *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);"""
PREFILL_ANCHOR = """                pfoff[b] += n;
                ds4_metric_add(&ds4_metrics_get()->tokens_prefilled_computed, n);"""
INJECT_ANCHOR = 'const int dspark_prefill_inject = dspark_mode && getenv("DS4_DSPARK_NO_PREFILL_INJECT") == NULL;'
TOKENIZER_ANCHOR = """static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    const uint64_t len = strlen(text);"""
SPECIAL_ANCHOR = "static bool special_token_at(const ds4_vocab *vocab, const char *p, int *token, size_t *len) {"


def prepare(source: Path, build_dir: Path):
    text = (source / "ds4.c").read_text()
    if text.count(ANCHOR) != 1:
        raise ValueError("pinned native sampler anchor changed; refusing an unverified hook")
    text = text.replace(
        ANCHOR,
        """        uint64_t    *rng) {
    if (tf_ds4_sampler) return tf_ds4_sampler(logits, (int)n_vocab, tf_ds4_sampler_ud);
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);""",
    )
    if text.count(PREFILL_ANCHOR) != 1:
        raise ValueError("pinned native prefill anchor changed; refusing an unverified hook")
    text = text.replace(
        PREFILL_ANCHOR,
        """                pfoff[b] += n;
                if (tf_ds4_checkpoint) tf_ds4_checkpoint(tf_ds4_checkpoint_ud, pfbase[b] + pfoff[b]);
                ds4_metric_add(&ds4_metrics_get()->tokens_prefilled_computed, n);""",
    )
    if text.count(INJECT_ANCHOR) != 1:
        raise ValueError("pinned native drafter prefill anchor changed; refusing an unverified hook")
    text = text.replace(
        INJECT_ANCHOR,
        "const int dspark_prefill_inject = (dspark_mode || (tf_ds4_checkpoint && dspark_armed)) "
        '&& getenv("DS4_DSPARK_NO_PREFILL_INJECT") == NULL;',
    )
    for name in ("pf_inj_pos", "pf_inj_sid"):
        anchor = f"int32_t *{name} = dspark_mode ?"
        if text.count(anchor) != 1:
            raise ValueError("pinned native drafter scratch anchor changed; refusing an unverified hook")
        text = text.replace(anchor, f"int32_t *{name} = dspark_prefill_inject ?")
    # A non-power-of-two context can end inside a capture band. The scorer
    # validates its scan against the allocated bank, even for masked rows.
    # Use the same bounded band for the graph key and the score/top-k stride.
    band_anchor = "/* C3-Inc2 twin selftest"
    band_counts = ("n_comp", "cap_ncomp[il]")
    band_calls = tuple(f"metal_graph_cont_capture_band({count})" for count in band_counts)
    if any(text.count(anchor) != 1 for anchor in (band_anchor, *band_calls)):
        raise ValueError("pinned native capture-band anchor changed; refusing an unverified hook")
    text = text.replace(
        band_anchor,
        """static uint32_t tf_ds4_capture_band(uint32_t count, uint32_t capacity) {
    const uint32_t band = metal_graph_cont_capture_band(count);
    return band < capacity ? band : capacity;
}

"""
        + band_anchor,
    )
    for call, count in zip(band_calls, band_counts):
        text = text.replace(call, f"tf_ds4_capture_band({count}, g->layer_comp_cap[il])")
    lookup = "static int vocab_lookup(const ds4_vocab *vocab, const char *text) {"
    if text.count(TOKENIZER_ANCHOR) != 1 or text.count(lookup) != 1:
        raise ValueError("pinned native tokenizer anchor changed; refusing an unverified hook")
    text = text.replace(
        TOKENIZER_ANCHOR,
        "static void tf_ds4_bpe_bytes(const ds4_vocab *vocab, const char *text, uint64_t len, token_vec *out) {",
    )
    text = text.replace(
        lookup,
        """static void bpe_tokenize_text(const ds4_vocab *vocab, const char *text, token_vec *out) {
    tf_ds4_bpe_bytes(vocab, text, strlen(text), out);
}

"""
        + lookup,
    )
    span = "static void tokenize_span(const ds4_vocab *vocab, const char *p, size_t n, token_vec *out) {"
    match = "if (!strncmp(p, specials[i].text, n)) {"
    if any(text.count(anchor) != 1 for anchor in (SPECIAL_ANCHOR, span, match)):
        raise ValueError("pinned native special-token anchor changed; refusing an unverified hook")
    start, end = text.index(SPECIAL_ANCHOR), text.index(span)
    bounded = (
        text[start:end]
        .replace(
            SPECIAL_ANCHOR,
            "static bool tf_ds4_special_bytes(const ds4_vocab *vocab, const char *p, size_t remaining, int *token, size_t *len) {",
        )
        .replace(match, "if (n <= remaining && !memcmp(p, specials[i].text, n)) {")
    )
    text = text[:start] + bounded + text[start:]
    (build_dir / "ds4-sampling.inc").write_text(text)
