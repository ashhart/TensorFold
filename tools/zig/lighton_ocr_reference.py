"""Pinned offline Transformers LightOnOCR-3 references: source pins, the model card example, HIP capture tensors."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import re
import sys
import time
from pathlib import Path

SCHEMA = "tensorfold.lighton-ocr.reference.v1"
EXAMPLES_SCHEMA = "tensorfold.lighton-ocr.examples.v2"
VERSION = "5.22.1"
CHECKPOINTS = {
    "0.8B": ("lightonai/LightOnOCR-3-0.8B", "36d8636925327c671bd8e479c517a004f7566bd9"),
    "1B": ("lightonai/LightOnOCR-3-1B", "23d544085c886bbdace647bcfab91203299417e2"),
    "4B": ("lightonai/LightOnOCR-3-4B", "b71010f095dc3735de7ee6b9969bff8f8f50b39a"),
}
# Every non-weight file of each pinned snapshot: card, config, tokenizer, template, processor, evals and logo.
SOURCES = {
    "0.8B": {
        ".eval_results/olmocrbench.yaml": "3141a24de395cf8137f9107530b407d65760f411271230a39d1f368d123e5b77",
        ".eval_results/parsebench.yaml": "25e4902aa6ba2d9a4dfdf05c8eb135c0a5af161281a58262fb5bb8129975327f",
        ".gitattributes": "09cc0ca3448f8d64e5d162edc582c0e264516b6566ae8aa19411be309333465d",
        "README.md": "0df44eab78278bad416d8974e96a6a181d59bae8416fa54877bd5b022e76aaf2",
        "chat_template.jinja": "273d8e0e683b885071fb17e08d71e5f2a5ddfb5309756181681de4f5a1822d80",
        "config.json": "842fafd8f06807269960515fa6f0e4b5dd27f9e645b67a2b3e84f1699ee8d60c",
        "generation_config.json": "b1cfebea5837fc69aa89c3e2210dbe6494bb29a04e1e88beb227ddad53b9dfc1",
        "lightonocr3_logo.png": "768eee79d378518cfea9ed8d81ba3a53a0cff17ccf4b12c22a9e5ebedde32bcf",
        "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "tokenizer.json": "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523",
        "tokenizer_config.json": "c908cd42c5e6eac721445fb70841e45845a6a269c6ef2c595e97a578a9724671",
        "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    },
    "1B": {
        ".eval_results/olmocrbench.yaml": "fa822c9b979aafcc4086d6e7f0c55081a2b282cf997bd560428df4d671feba00",
        ".eval_results/parsebench.yaml": "c5433b13d1834f6d233d85bee5296e2f92e92196fee5ee7cfffe1f63feb895aa",
        ".gitattributes": "464a5728a36040d107bc2168d8907487737c3912b541cba91fde29887b123dd6",
        "README.md": "739f34999e8fd672b56ed73d966d6afeee77ff8a0cea641501b6d3133e02e834",
        "chat_template.jinja": "0942a43ee0ba48a608814e0b40ac9cb70bcc29ea72f19c84b1bf2597f05aa2bd",
        "config.json": "43b62f9ec0a7207f67fd032f7ef9719fa8b6b44d7b3cf20fbc2346329493ff3a",
        "generation_config.json": "7d2566b271454af41e9ebc28851888e1ec7ca33a53fe644dac207856df952981",
        "lightonocr3_logo.png": "768eee79d378518cfea9ed8d81ba3a53a0cff17ccf4b12c22a9e5ebedde32bcf",
        "processor_config.json": "5292a61a089835ec6884836e3896bba1fc571c60df872e3ca9eb45bd3288910d",
        "special_tokens_map.json": "76862e765266b85aa9459767e33cbaf13970f327a0e88d1c65846c2ddd3a1ecd",
        "tokenizer.json": "f54b55fa0c3aba0c91ce09ea79ae4e62da24e2a1a630f96c4bae34aba25e234a",
        "tokenizer_config.json": "76d82c8010d0e0d5431aa5e3658c69e11a2a7250e789799d2be4ab75dc3fafed",
    },
    "4B": {
        ".eval_results/olmocrbench.yaml": "a2f531379cac28191fabeb2068f5942176f7dad89c48518cc88ac9152b61fb3e",
        ".eval_results/parsebench.yaml": "f6e6ffbbb0e092462d268b682f8eb408e86cd7a743bbf7f252b3ba376bb1bd3f",
        ".gitattributes": "464a5728a36040d107bc2168d8907487737c3912b541cba91fde29887b123dd6",
        "README.md": "4c56af1be12a620d0cadbb09976c0cd02fac5557f0f780be747d35bd6689fc54",
        "README.no_thinking.md": "8c909ee2086b401f6c6bfa016f43ca72777046b8e0ee8f496c898a8cd80b4538",
        "chat_template.jinja": "d5ddb788d5f3df12e32244954037eb535409d24eeb48c2429d7588af4e7b69a7",
        "config.json": "99b4c363a998c2d8c5ed5b3f3a40ec5cb97a156034df5002d0d789c74324f5f0",
        "generation_config.json": "8ce51af0baac298c116b7db144efe1ebd78b0b792d084002634fd3ed63f4d515",
        "lightonocr3_logo.png": "768eee79d378518cfea9ed8d81ba3a53a0cff17ccf4b12c22a9e5ebedde32bcf",
        "merges.txt": "a9d356d7bdf1ef4949e3e748e95b8e10ad9d4e2e838eddc38a0a7b6b94d1db8d",
        "preprocessor_config.json": "27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516",
        "processor_config.json": "d89ef49ce9cd37fbf510158e13c1ef063d9286411c1ec9049932dbe0487143b1",
        "tokenizer.json": "87a7830d63fcf43bf241c3c5242e96e62dd3fdc29224ca26fed8ea333db72de4",
        "tokenizer_config.json": "e98f1901ac6f0adff67b1d540bfa0c36ac1a0cf59eb72ed78146ef89aafa1182",
        "vocab.json": "ce99b4cb2983d118806ce0a8b777a35b093e2000a503ebde25853284c9dfa003",
    },
}
# Transformers model code per size, package-relative: Qwen3.5 for 0.8B/4B; LightOnOcr (Pixtral + Qwen3) for 1B.
QWEN35_LIBRARY = {
    "models/qwen2_vl/image_processing_qwen2_vl.py":
        "ebdb484798a6aefebc5dbe5431d3e4af8f460404d95d4cb67165fdff624ed333",
    "models/qwen3_5/configuration_qwen3_5.py": "332ea31ed31cc81ddb72da28737e9868a55d295f9f1daa3e0ad0d57d56d4e657",
    "models/qwen3_5/modeling_qwen3_5.py": "af05d0acfe79fff8aea1c75f93516a6a72bc80f6f901329a73e66a3866fe1e7c",
    "models/qwen3_5/tokenization_qwen3_5.py": "3d2f98abc1c6eec61e4548171286b88f5e8173ac6d63954b8fd225b7e24cf25a",
    "models/qwen3_vl/processing_qwen3_vl.py": "02d50224d9dc9ce38690cfa6a0106be897f619444ebe3ff7ba38cbcdda9e3ef2",
    "models/qwen3_vl/video_processing_qwen3_vl.py":
        "1d7be665171c8441108b43f179759fa39a35a5a132c14a142a296fc295299b7a",
}
LIGHTON_LIBRARY = {
    "models/lighton_ocr/configuration_lighton_ocr.py":
        "54d5c0ae3816653015cec91f756233741424efcb06d21d9e487d28312f6b30d7",
    "models/lighton_ocr/modeling_lighton_ocr.py": "b7419e43defa022b3edf1a17e121e561678043f8f86ecebcb765a7e1e1ddb8e2",
    "models/lighton_ocr/processing_lighton_ocr.py":
        "3bd0e19f778809931abe45ee1bc66460b77d9651d532364768ca2c12305662a0",
    "models/pixtral/configuration_pixtral.py": "254fc400b064716d3c25b9cefec0b9872f947bf7ccf408c15275a385a2827fdd",
    "models/pixtral/image_processing_pixtral.py": "9e7804a295421d91e61c81d501e4d2b0cfead88cbee6a092afc1f1fff23888f8",
    "models/pixtral/modeling_pixtral.py": "503e23525215ae1e59e32e1132159502b431003218874d185da9ee6842775099",
    "models/qwen3/configuration_qwen3.py": "49d9ccd0f29ccd977b93dba2004f84b867891b3d0509b71fd8ba3ed01054e64d",
    "models/qwen3/modeling_qwen3.py": "63a6213c15d6fb04e58a1c64fd05a4cdfb49476443fbeef524f0e0486043824b",
}
LIBRARY = {"0.8B": QWEN35_LIBRARY, "1B": LIGHTON_LIBRARY, "4B": QWEN35_LIBRARY}
# tree_sha256 of the whole installed Transformers 5.22.1 package, so shared code is pinned too.
TRANSFORMERS_TREE = "b08cead6d1b841ef9723106a81a92ec37d67aff3e43a198e56b7dc76538d377b"
# Each card's template and generate() kwargs; the 1B samples from generation_config.json (T 0.2, top_p 0.9, top_k 0).
CARD = {
    "0.8B": ({"enable_thinking": False}, {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.2}),
    "1B": ({}, {"max_new_tokens": 1024}),
    "4B": ({"enable_thinking": False}, {"max_new_tokens": 1024, "do_sample": True, "temperature": 0.2}),
}
RECEIPT_URL = ("https://huggingface.co/datasets/hf-internal-testing/fixtures_ocr/resolve/"
               "28fe12cdf7816b5dde94e22051b2ec8dc74267b7/SROIE-receipt.jpeg")
RECEIPT_SHA256 = "d66a3287c847823c6ddab9f8b2080582b96a66b7816d178472f697ea84e227b3"
ASSETS = (
    "config.json", "processor_config.json", "preprocessor_config.json", "tokenizer_config.json",
    "tokenizer.json", "special_tokens_map.json", "chat_template.jinja", "generation_config.json",
)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hashed(root: Path, pins: dict, what: str) -> dict:
    """Return the hashes of the pinned files under root, or refuse when any differs from its pin."""
    found = {name: digest(root / name) if (root / name).is_file() else None for name in pins}
    if wrong := sorted(name for name, value in found.items() if value != pins[name]):
        raise ValueError(f"{what} differs from its pin: {', '.join(wrong)}")
    return found


def tree_sha256(package: Path) -> str:
    """Hash every file of an installed package (bytecode caches excluded) as sorted "path NUL sha256" lines."""
    # rglob does not descend into directory links, and a linked directory could shadow a pinned module.
    if links := sorted(path.relative_to(package).as_posix() for path in package.rglob("*")
                       if path.is_symlink() and path.is_dir()):
        raise ValueError(f"{package.name} has linked directories: {', '.join(links)}")
    files = sorted(path for path in package.rglob("*")
                   if path.is_file() and "__pycache__" not in path.relative_to(package).parts)
    lines = (f"{path.relative_to(package).as_posix()}\0{digest(path)}\n" for path in files)
    return hashlib.sha256("".join(lines).encode()).hexdigest()


def pinned(model: Path, size: str) -> dict:
    """Refuse, before any model code runs, a snapshot or Transformers install whose sources differ from the pins."""
    repo, revision = CHECKPOINTS[size]
    model = model.resolve()
    if model.parent.name == "snapshots" and model.name != revision:
        raise ValueError(f"{size}: snapshot directory is not the pinned revision {revision}")
    # rglob does not descend into directory links, so a linked directory could hide files Transformers reads.
    if links := sorted(path.relative_to(model).as_posix() for path in model.rglob("*")
                       if path.is_symlink() and path.is_dir()):
        raise ValueError(f"{size}: snapshot has linked directories: {', '.join(links)}")
    extra = sorted(path.relative_to(model).as_posix() for path in model.rglob("*")
                   if path.is_file() and path.suffix != ".safetensors"
                   and path.relative_to(model).as_posix() not in SOURCES[size])
    if extra:
        raise ValueError(f"{size}: snapshot has unpinned non-weight files: {', '.join(extra)}")
    if (version := importlib.metadata.version("transformers")) != VERSION:
        raise ValueError(f"expected Transformers {VERSION}, found {version}")
    package = Path(importlib.util.find_spec("transformers").origin).parent
    # The named files are the family's own code; the tree pin also covers the shared processing and generation code.
    if (tree := tree_sha256(package)) != TRANSFORMERS_TREE:
        raise ValueError(f"Transformers {VERSION} package tree differs from its pin: {tree}")
    return {"checkpoint": hashed(model, SOURCES[size], f"{repo}@{revision}"),
            "transformers": hashed(package, LIBRARY[size], f"Transformers {VERSION}"),
            "transformers_tree": tree}


def sources(args) -> dict:
    """Check the snapshot and the Transformers sources against the pins and return their hashes; no model loads."""
    return {"repo": CHECKPOINTS[args.size][0], "revision": CHECKPOINTS[args.size][1], "transformers": VERSION,
            "source_sha256": pinned(args.model, args.size)}


def ids_sha256(ids: list[int]) -> dict:
    """Hash token ids as comma-joined ASCII decimals (token_sha256) and as little-endian u32 words."""
    return {"token_sha256": hashlib.sha256(",".join(map(str, ids)).encode("ascii")).hexdigest(),
            "u32le_sha256": hashlib.sha256(b"".join(i.to_bytes(4, "little") for i in ids)).hexdigest()}


def checkpoint(model: Path, size: str, revision: str) -> dict:
    """Bind the caller's local snapshot to a specific published revision and immutable local asset digests."""
    repo, expected = CHECKPOINTS[size]
    if revision != expected:
        raise ValueError(f"{size}: expected pinned revision {expected}, got {revision}")
    model = model.resolve()
    if model.parent.name == "snapshots" and model.name != revision:
        raise ValueError("local snapshot directory does not match --revision")
    config = json.loads((model / "config.json").read_text())
    expected_type = "mistral3" if size == "1B" else "qwen3_5"
    architecture = "LightOnOCRForConditionalGeneration" if size == "1B" else "Qwen3_5ForConditionalGeneration"
    width = {"0.8B": 1024, "1B": 1024, "4B": 2560}[size]
    if (config.get("model_type") != expected_type or architecture not in config.get("architectures", [])
            or config.get("text_config", {}).get("hidden_size") != width):
        raise ValueError("local configuration does not match the selected checkpoint")
    assets = {name: digest(model / name) for name in ASSETS if (model / name).is_file()}
    if "processor_config.json" not in assets or "tokenizer.json" not in assets:
        raise ValueError("local processor or tokenizer assets are incomplete")
    weights = sorted(model.glob("*.safetensors"))
    if not weights:
        raise ValueError("local safetensors weights are missing")
    index = model / "model.safetensors.index.json"
    if index.exists():
        assets[index.name] = digest(index)
        if set(json.loads(index.read_text())["weight_map"].values()) != {path.name for path in weights}:
            raise ValueError("local safetensors shards do not match the checkpoint index")
    weights_sha256 = {path.name: digest(path) for path in weights}
    return {"repo": repo, "revision": revision, "assets": assets, "weights_sha256": weights_sha256}


def read_ids(path: Path | None, vocab: int) -> list[int]:
    if path is None:
        return []
    raw = path.read_text()
    values = json.loads(raw) if raw.lstrip().startswith("[") else [int(x) for x in raw.split()]
    if not isinstance(values, list) or any(type(x) is not int or not 0 <= x < vocab for x in values):
        raise ValueError("continuation must contain integer token IDs inside the vocabulary")
    return values


def save_array(folder: Path, tensors: dict, name: str, values, source_dtype: str | None = None) -> None:
    """Write one little-endian <f4 or <i8 .npy file and the manifest record that fixture.zig verifies."""
    import numpy as np

    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name) or name in tensors:
        raise ValueError(f"duplicate or invalid tensor name {name!r}")
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or not array.size or not np.isfinite(array).all():
        raise ValueError(f"{name}: expected a nonempty finite numeric tensor")
    dtype = np.dtype("<f4") if array.dtype.kind == "f" else np.dtype("<i8")
    if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
        raise ValueError(f"{name}: integer does not fit int64")
    array = np.ascontiguousarray(array, dtype=dtype)
    path = folder / f"{name}.npy"
    np.save(path, array, allow_pickle=False)
    tensors[name] = {"file": path.name, "shape": list(array.shape), "dtype": array.dtype.str,
                     "source_dtype": source_dtype or str(array.dtype), "sha256": digest(path)}


def top_rows(values) -> list[dict]:
    import numpy as np

    if values.ndim != 2 or values.shape[1] < 2 or not np.isfinite(values).all():
        raise ValueError("logits must be finite [rows, vocabulary>=2]")
    result = []
    for row in values:
        first = int(row.argmax())
        other = row.copy()
        other[first] = -np.inf
        second = int(other.argmax())
        logits = [float(row[first]), float(row[second])]
        result.append({"ids": [first, second], "logits": logits, "margin": logits[0] - logits[1]})
    return result


def attention_manifest(model, requested: str) -> dict:
    """Read loaded component configurations rather than assuming propagation from the loader argument."""
    base = model.model
    tower = base.vision_encoder if hasattr(base, "vision_encoder") else base.visual
    configs = {"root": model.config, "text": base.language_model.config, "vision": tower.config}
    actual = {key: getattr(config, "_attn_implementation", None) for key, config in configs.items()}
    if any(not isinstance(actual[key], str) for key in ("text", "vision")):
        raise ValueError("loaded model does not expose both actual attention implementations")
    linear = "linear_attention" in getattr(base.language_model.config, "layer_types", ())
    return {"requested": requested, "actual": actual,
            "comparison": "use independent numerical bounds; eager reference does not imply native byte identity",
            "linear_attention": "checkpoint-defined GDN; attention setting describes full-attention layers"
            if linear else "none"}


def source_manifest(transformers, torch, objects: list, expected_version: str, model, attention: str) -> dict:
    """Record installed code, not an unverified caller-supplied source commit."""
    version = importlib.metadata.version("transformers")
    if version != expected_version:
        raise ValueError(f"installed transformers {version} differs from expected {expected_version}")
    root = Path(transformers.__file__).resolve().parent
    files = {}
    for obj in objects:
        module = inspect.getmodule(type(obj))
        path = Path(module.__file__).resolve() if module is not None and module.__file__ else None
        if path and path.is_relative_to(root):
            files[str(path.relative_to(root))] = digest(path)
    direct = importlib.metadata.distribution("transformers").read_text("direct_url.json")
    commit = (json.loads(direct).get("vcs_info", {}) if direct else {}).get("commit_id")
    return {"library": "https://github.com/huggingface/transformers", "version": version, "source_revision": commit,
            "release_tag": None if commit else f"v{version}",
            "revision_basis": "installed-vcs-commit" if commit else "installed-version-and-file-digests",
            "source_files_sha256": files, "torch": torch.__version__, "cuda": torch.version.cuda,
            "hip": torch.version.hip, "attention": attention_manifest(model, attention), "use_cache": False}


def load_reference(model_path: Path, size: str, torch, transformers, dtype, attention="eager"):
    """Explicit classes bypass legacy top-level mistral3 dispatch; the local checkpoint is never rewritten."""
    if size == "1B":
        payload = json.loads((model_path / "config.json").read_text())
        payload["model_type"] = "lighton_ocr"
        payload["architectures"] = ["LightOnOcrForConditionalGeneration"]
        model_cls = transformers.LightOnOcrForConditionalGeneration
        processor_cls = transformers.LightOnOcrProcessor
        extra = {"config": transformers.LightOnOcrConfig.from_dict(payload),
                 "key_mapping": {r"^language_model\.model\.": "model.language_model."}}
    else:
        model_cls, processor_cls, extra = transformers.Qwen3_5ForConditionalGeneration, transformers.AutoProcessor, {}
    processor = processor_cls.from_pretrained(model_path, local_files_only=True, trust_remote_code=False)
    model, loading = model_cls.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, dtype=dtype,
                                               attn_implementation=attention, output_loading_info=True, **extra)
    keys = ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")
    errors = {key: loading[key] for key in keys if loading.get(key)}
    if errors:
        raise ValueError(f"reference checkpoint did not load completely: {errors}")
    if (model.config.text_config.tie_word_embeddings
            and model.get_input_embeddings().weight.data_ptr() != model.get_output_embeddings().weight.data_ptr()):
        raise ValueError("reference did not tie its input and output embeddings")
    return model.eval(), processor


def add_continuation(batch: dict, ids: list[int], torch) -> dict:
    result = dict(batch)
    if not ids:
        return result
    result["input_ids"] = torch.cat([batch["input_ids"], torch.tensor([ids], dtype=batch["input_ids"].dtype)], dim=1)
    for name in ("attention_mask", "token_type_ids", "mm_token_type_ids"):
        if name in result:
            fill = 1 if name == "attention_mask" else 0
            result[name] = torch.cat([batch[name], batch[name].new_full((1, len(ids)), fill)], dim=1)
    if "position_ids" in result:
        raise ValueError("explicit processor position_ids with continuation need a verified position-extension rule")
    return result


def capture(args) -> dict:
    source_sha256 = pinned(args.model, args.size)
    if args.output.exists():
        raise ValueError("output already exists; use a new directory to avoid mixing fixtures")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("capture output must be outside the source checkout")
    if importlib.metadata.version("transformers") != args.expect_transformers_version:
        raise ValueError("installed transformers version does not match --expect-transformers-version")
    import torch
    import transformers
    from PIL import Image

    info = checkpoint(args.model, args.size, args.revision)
    config = json.loads((args.model / "config.json").read_text())
    ids = read_ids(args.continuation, config["text_config"]["vocab_size"])
    torch.manual_seed(0)
    torch.set_float32_matmul_precision("highest")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        torch.backends.cudnn.allow_tf32 = False
    dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[args.dtype]
    model, processor = load_reference(args.model, args.size, torch, transformers, dtype, args.attention)
    model.to(args.device)
    images = []
    for path in args.image:
        with Image.open(path) as opened:
            images.append(opened.convert("RGB"))
    content = [{"type": "image"} for _ in images]
    if args.mode == "grounding":
        content.append({"type": "text", "text": "grounding"})
    rendered = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                             tokenize=False, enable_thinking=False)
    batch = dict(processor(text=[rendered], images=images, return_tensors="pt"))
    prompt_ids = batch["input_ids"][0].tolist()
    args.output.mkdir(parents=True)
    tensors, hooks = {}, []

    def save(name, tensor):
        if not torch.is_tensor(tensor):
            raise ValueError(f"{name}: reference hook did not return a tensor")
        value = tensor.detach().cpu()
        array = value.float().numpy() if value.is_floating_point() else value.numpy()
        save_array(args.output, tensors, name, array, str(tensor.dtype))

    def on_device(inputs: dict) -> dict:
        def move(value):
            return value.to(device=args.device, dtype=dtype if value.is_floating_point() else value.dtype)

        return {key: move(value) if torch.is_tensor(value) else value for key, value in inputs.items()}

    # These tensor names are the fixture contract that zig/tests/lighton_hip/*.zig reads.
    for name, tensor in batch.items():
        if torch.is_tensor(tensor):
            save("processor." + name, tensor)
    inputs = on_device(add_continuation(batch, ids, torch))
    for name, tensor in inputs.items():
        if torch.is_tensor(tensor):
            save("teacher." + name, tensor)
    base = model.model
    if args.size == "1B":
        projector = base.vision_projection
        encoder = base.vision_encoder
        hooks.append(encoder.register_forward_hook(lambda _m, _a, out: save("vision.encoder", out.last_hidden_state)))
    else:
        projector = base.visual.merger
        hooks.append(projector.register_forward_pre_hook(lambda _m, positional: save("vision.encoder", positional[0])))
    hooks.append(projector.register_forward_hook(lambda _m, _a, out: save("vision.projector", out)))

    def language_inputs(_module, _positional, kwargs):
        for name in ("inputs_embeds", "position_ids"):
            if kwargs.get(name) is not None:
                save("decoder." + name, kwargs[name])

    hooks.append(base.language_model.register_forward_pre_hook(language_inputs, with_kwargs=True))
    try:
        with torch.inference_mode():
            output = model(**inputs, use_cache=False, logits_to_keep=len(ids) + 1)
            logits = output.logits[0, -(len(ids) + 1) :]
            save("logits", logits)
    finally:
        for hook in hooks:
            hook.remove()
    required = {"vision.encoder", "vision.projector", "decoder.inputs_embeds", "logits"}
    if not required.issubset(tensors):
        raise ValueError(f"missing reference stages: {sorted(required - tensors.keys())}")
    if args.generate:
        with torch.inference_mode():
            generated = model.generate(**on_device(batch), do_sample=False, use_cache=False,
                                       max_new_tokens=args.generate)
        save("generated.ids", generated[0, len(prompt_ids) :])
    objects = [processor, processor.image_processor, *model.modules()]
    manifest = {
        "schema": SCHEMA,
        "checkpoint": info,
        "source_sha256": source_sha256,
        "request": {"mode": args.mode, "images_sha256": [digest(path) for path in args.image],
                    "prompt_ids": prompt_ids, "continuation_ids": ids, "generate": args.generate},
        "reference": source_manifest(transformers, torch, objects, args.expect_transformers_version, model,
                                     args.attention),
        "settings": {"dtype": args.dtype, "device": args.device, "tf32": False,
                     "bf16_reduced_precision_reduction": False, "seed": 0},
        "teacher_rows": {"prompt_length": len(prompt_ids),
                         "positions": list(range(len(prompt_ids) - 1, len(prompt_ids) + len(ids))),
                         "targets": ids + [None]},
        "logit_top2": top_rows(logits.detach().float().cpu().numpy()),
        "tensors": tensors,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return {"output": str(args.output), "tensors": len(tensors), "prompt_tokens": len(prompt_ids),
            "teacher_rows": len(ids) + 1, "qualification": "Transformers reference capture only"}


def examples(args) -> dict:
    """Run the model card's Transformers example: greedy, and the card's own sampled generate() call."""
    source_sha256 = pinned(args.model, args.size)
    if args.output.exists():
        raise ValueError("output already exists")
    if args.output.resolve().is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("examples output must be outside the source checkout")
    if digest(args.image) != RECEIPT_SHA256:
        raise ValueError(f"image is not the model card's receipt {RECEIPT_URL}")
    import PIL
    import torch
    import transformers
    from transformers.image_utils import load_image

    template, card = CARD[args.size]
    info = checkpoint(args.model, args.size, CHECKPOINTS[args.size][1])
    receipt = {
        "schema": EXAMPLES_SCHEMA, "status": "running", "card": f"https://huggingface.co/{info['repo']}/blob/"
        f"{info['revision']}/README.md", "checkpoint": info, "source_sha256": source_sha256,
        "image": {"url": RECEIPT_URL, "sha256": RECEIPT_SHA256},
        "runtime": {"python": platform.python_version(), "transformers": transformers.__version__,
                    "torch": torch.__version__, "hip": torch.version.hip, "cuda": torch.version.cuda,
                    "pillow": PIL.__version__, "tokenizers": importlib.metadata.version("tokenizers"),
                    "device": args.device, "device_name": None},
        "settings": {"dtype": "bfloat16", "attention": args.attention, "seed": args.seed,
                     "chat_template_kwargs": template, "image_decode": "transformers.image_utils.load_image (Pillow)"},
        "runs": [],
    }

    def write():
        args.output.write_text(json.dumps(receipt, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    modes = ("grounding", "transcription") if args.mode == "both" else (args.mode,)
    methods = ("greedy", "sampled") if args.sampling == "both" else (args.sampling,)
    try:
        receipt["runtime"]["device_name"] = (torch.cuda.get_device_name(args.device)
                                             if torch.device(args.device).type == "cuda" else platform.processor())
        model, processor = load_reference(args.model, args.size, torch, transformers, torch.bfloat16, args.attention)
        model.to(args.device)
        defaults = model.generation_config
        receipt["settings"]["generation_config"] = {key: getattr(defaults, key) for key in (
            "do_sample", "temperature", "top_k", "top_p", "use_cache", "eos_token_id", "pad_token_id")}
        # The card's URL needs a JPEG-capable torchvision; Transformers' own Pillow decode gives the same RGB image.
        image = load_image(str(args.image))
        for mode in modes:
            content = [{"type": "image", "image": image}]
            if mode == "grounding":
                content.append({"type": "text", "text": "grounding"})
            inputs = processor.apply_chat_template([{"role": "user", "content": content}], add_generation_prompt=True,
                                                   tokenize=True, return_dict=True, return_tensors="pt", **template)
            # The 1B card casts floating inputs to the model dtype; Qwen3.5 casts pixel_values to its tower dtype.
            inputs = {key: value.to(device=args.device, dtype=torch.bfloat16 if value.is_floating_point()
                                    else value.dtype) if torch.is_tensor(value) else value
                      for key, value in inputs.items()}
            prompt = inputs["input_ids"][0].tolist()
            for method in methods:
                kwargs = {**card, "max_new_tokens": args.max_new_tokens} if method == "sampled" else {
                    "max_new_tokens": args.max_new_tokens, "do_sample": False}
                torch.manual_seed(args.seed)
                began = time.monotonic()
                with torch.inference_mode():
                    ids = model.generate(**inputs, **kwargs)[0, len(prompt) :].tolist()
                receipt["runs"].append({
                    "mode": mode, "sampling": method, "generate_kwargs": kwargs, "seconds": time.monotonic() - began,
                    "prompt": {"length": len(prompt), **ids_sha256(prompt), "ids": prompt},
                    "output": {"length": len(ids), **ids_sha256(ids), "ids": ids,
                               "text": processor.decode(ids, skip_special_tokens=True)},
                })
                write()
    except Exception as error:
        receipt.update(status="failed", error=f"{type(error).__name__}: {error}")
        write()
        raise
    receipt["status"] = "complete"
    write()
    return {"output": str(args.output), "runs": [
        {"mode": run["mode"], "sampling": run["sampling"], "prompt_tokens": run["prompt"]["length"],
         "output_tokens": run["output"]["length"], "token_sha256": run["output"]["token_sha256"]}
        for run in receipt["runs"]]}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="command", required=True)
    pins = modes.add_parser("sources", help="check a local snapshot and Transformers against the pins; no model loads")
    pins.add_argument("--model", type=Path, required=True, help="local snapshot directory")
    pins.add_argument("--size", choices=CHECKPOINTS, required=True)
    pins.set_defaults(func=sources)
    card = modes.add_parser("examples", help="run the model card's Transformers example and write a JSON receipt")
    card.add_argument("--model", type=Path, required=True, help="local snapshot directory")
    card.add_argument("--size", choices=CHECKPOINTS, required=True)
    card.add_argument("--image", type=Path, required=True, help=f"local copy of {RECEIPT_URL}")
    card.add_argument("--output", type=Path, required=True, help="new JSON receipt outside the source checkout")
    card.add_argument("--device", required=True, help="explicit PyTorch device, such as cpu or cuda:0")
    card.add_argument("--mode", choices=("grounding", "transcription", "both"), default="both")
    card.add_argument("--sampling", choices=("greedy", "sampled", "both"), default="both")
    card.add_argument("--attention", choices=("sdpa", "eager"), default="sdpa", help="the card loads the default sdpa")
    card.add_argument("--max-new-tokens", type=int, default=1024, help="the cards use 1024")
    card.add_argument("--seed", type=int, default=0)
    card.set_defaults(func=examples)
    cap = modes.add_parser("capture", help="local checkpoint and local images to reference .npy tensors")
    cap.set_defaults(func=capture)
    cap.add_argument("--model", type=Path, required=True)
    cap.add_argument("--size", choices=CHECKPOINTS, required=True)
    cap.add_argument("--revision", required=True)
    cap.add_argument("--expect-transformers-version", required=True)
    cap.add_argument("--image", type=Path, action="append", required=True)
    cap.add_argument("--output", type=Path, required=True, help="new fixture directory outside the source checkout")
    cap.add_argument("--device", required=True, help="explicit PyTorch device, such as cpu or cuda:0")
    cap.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    cap.add_argument("--attention", choices=("eager", "sdpa"), default="eager")
    cap.add_argument("--mode", choices=("transcription", "grounding"), default="transcription")
    cap.add_argument("--continuation", type=Path, help="JSON list or whitespace-separated teacher token IDs")
    cap.add_argument("--generate", type=int, default=0, help="also capture this many greedy output tokens")
    args = parser.parse_args(argv)
    try:
        if not 0 <= getattr(args, "generate", 0) <= 16384 or not 1 <= getattr(args, "max_new_tokens", 1) <= 16384:
            raise ValueError("--generate must be in 0..16384 and --max-new-tokens in 1..16384")
        print(json.dumps(args.func(args), indent=2, allow_nan=False))
        return 0
    except (ValueError, TypeError, OSError, ImportError, importlib.metadata.PackageNotFoundError) as error:
        print(f"lighton reference: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
