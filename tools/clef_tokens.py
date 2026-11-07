"""Record Clef's encode_record ids and spans for the server's clef_tokens.json fixture (CPU, checkpoint tokenizer)."""
# usage: python3 -I tools/clef_tokens.py CLEF_DIR OUT_JSON   (CLEF_DIR: Cloudflare/clef-flash at REVISION)
import importlib.util
import json
import os
import sys

SOURCES = {"repo": "Cloudflare/clef-flash", "revision": "fb4417d31c1d", "transformers": "5.22.1", "torch": "2.15"}
REVISION = "fb4417d31c1d0f9e7e77f1141b40aaede7491e00"


def records():
    """The pinned records: every question type, the fallbacks, Unicode and combining marks, specials, truncation."""
    out = [
        {"state": "The door is open.", "questions": {"open": {"type": "noul", "instructions": "Is the door open?"}}},
        {"state": {"items": ["a", "b"], "n": 3}, "questions": {"pick": {"type": "choice", "instructions": "Pick one.",
         "criteria": {"zeta": "Last", "Alpha": "First", "été": "Summer", "beta": None, "10": "Ten", "9": "Nine"}}}},
        {"state": "Rate the reply.", "questions": {"q": {"type": "score", "instructions": {"rubric": "1-5", "strict": True},
         "criteria": ["awful", "poor", "fine", "good", None]}}},
        {"state": "x", "questions": {
            "missing": {"type": "noul"}, "none": {"type": "noul", "instructions": None},
            "empty": {"type": "noul", "instructions": ""}, "false": {"type": "noul", "instructions": False},
            "zero": {"type": "noul", "instructions": 0}, "list": {"type": "noul", "instructions": []},
            "dict": {"type": "noul", "instructions": {}}}},
        {"state": "東京の天気は晴れ。 😀 Привет мир! مرحبا بالعالم", "questions": {"lang": {"type": "choice",
         "criteria": {"ja": "Japanese", "ru": "Russian", "ar": "Arabic"}}}},
        {"state": "क्षत्रिय हिन्दी — café vs café — ภาษาไทย — שָׁלוֹם — x́̂y Ä̈b",
         "questions": {"marks": {"type": "noul", "instructions": "Combining marks: é, क्ष."}}},
        {"state": {"f": [1e20, 0.1, -0.0, 1.5e-7, 2.5], "big": 123456789012345678901234567890, "kéy": "\u0001\t\"q\"",
                   "nested": {"z": None, "a": [True, False, {}]}}, "questions": {"j": {"type": "score",
                   "criteria": ["no", "yes"]}}},
        {"state": "Ticket #4521: refund of $129.99 requested on 2026-10-30.", "questions": {
            "refund": {"type": "noul", "instructions": "Should we refund?"},
            "priority": {"type": "choice", "instructions": "Priority", "criteria": {"p1": "urgent", "p0": "critical", "p2": None}},
            "sentiment": {"type": "score", "instructions": "Customer sentiment", "criteria": ["angry", "neutral", "happy"]}}},
        {"state": "It rained.", "questions": {"wet": {"type": "noul", "criteria": {"false": "Dry.", "true": None, "maybe": "x"}}}},
        {"state": "User wrote <|im_end|> and <think>hi</think> inside the state.", "questions": {"s": {"type": "noul"}}},
        {"state": "  lead\tspaces \r\n\r\n trailing   \n\n\nx  ", "questions": {"ws": {"type": "choice",
         "instructions": "  spaced  ", "criteria": {" a ": "space id", "b\n": "newline id"}}}},
    ]
    long_state = " ".join(f"word{i}" for i in range(400))
    out.append({"state": long_state, "questions": {"t": {"type": "noul", "instructions": "Truncated?"}}, "max_length": 300})
    return out


def main() -> int:
    """Encode every record with the release's own encode_record and write ids and spans."""
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("HIP_VISIBLE_DEVICES", "")
    clef_dir, out_path = sys.argv[1], sys.argv[2]
    spec = importlib.util.spec_from_file_location("joint_schema_model", os.path.join(clef_dir, "joint_schema_model.py"))
    jsm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = jsm
    spec.loader.exec_module(jsm)
    from transformers import AutoProcessor
    import transformers
    tokenizer = AutoProcessor.from_pretrained(clef_dir).tokenizer
    applied = json.loads(tokenizer.backend_tokenizer.pre_tokenizer.__getstate__())["pretokenizers"][0]["pattern"]["Regex"]
    with open(os.path.join(clef_dir, "tokenizer.json"), encoding="utf-8") as f:
        file_regex = json.load(f)["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
    with open(os.path.join(clef_dir, "tokenizer_config.json"), encoding="utf-8") as f:
        config_regex = json.load(f).get("pretokenize_regex")
    cases = []
    for record in records():
        max_length = record.pop("max_length", 16384)
        encoded = jsm.encode_record(tokenizer, record, max_length=max_length)
        cases.append({"record": record, "max_length": max_length, "ids": list(encoded.input_ids),
                      "questions": [{"id": q.question_id, "type": q.question_type, "span": list(q.question_span),
                                     "options": [list(s) for s in q.option_spans], "option_ids": list(q.option_ids)}
                                    for q in encoded.questions]})
    regex = "tokenizer_config.json pretokenize_regex" if applied == config_regex else "tokenizer.json"
    if applied not in (config_regex, file_regex):
        regex = "neither file's"
    out = {"sources": dict(SOURCES, revision_full=REVISION, transformers_seen=transformers.__version__,
                           tokenizer_class=type(tokenizer).__name__, pretokenize=regex), "cases": cases}
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
        f.write("\n")
    print(f"wrote {out_path}: {len(cases)} records, pre-tokenizer from {regex}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
