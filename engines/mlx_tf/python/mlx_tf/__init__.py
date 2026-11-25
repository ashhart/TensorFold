"""TensorFold `mlx_tf` — Universal Bare-Metal Apple Silicon LLM & VLM Engine (Zero Apple MLX).

Drop-in replacement for `mlx_lm` backed directly by `libmlxtf.dylib` (Zig + handwritten Metal):
  - Zero Apple MLX imported or required
  - Universal Apple Silicon M1 through M5 (auto-selects M5 tensor ops or M1-M4 simdgroup GEMM)
  - Supports Native TFQ3 (3/3.5/4/5-bit + built-in Metal GPU quantizer), MLX Affine (2..8-bit, g32/64/128,
    mixed precision), and Lossless BF16/FP16
  - Supports Qwen3.5/3.8 (Dense & MoE), Qwen2/2.5/3 (Dense & MoE), Llama 3/3.1/3.2/3.3, Mistral/Mixtral,
    Gemma 2/3/4, DeepSeek-V2/V3/R1/V3.2 (MLA), GLM-4/4.5/4.7/5, LiquidAI LFM2/LFM2.5/LFM2-MoE,
    NVIDIA Nemotron-H/3.5, plus VLM Vision Encoders (Qwen3.5-VL & Gemma 3-VL SigLIP)
"""
from __future__ import annotations

import ctypes
import glob
import json
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

__version__ = "0.4.0"


class _MlxTfGenOptions(ctypes.Structure):
    _fields_ = [
        ("max_tokens", ctypes.c_uint32),
        ("temperature", ctypes.c_float),
        ("top_p", ctypes.c_float),
        ("top_k", ctypes.c_uint32),
        ("min_p", ctypes.c_float),
        ("seed", ctypes.c_uint64),
        ("enable_thinking", ctypes.c_uint8),
        ("use_chat_template", ctypes.c_uint8),
        ("stop_on_eos", ctypes.c_uint8),
        ("_pad", ctypes.c_uint8),
        ("repetition_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("_reserved", ctypes.c_uint32),
    ]


class _MlxTfStats(ctypes.Structure):
    _fields_ = [
        ("prompt_tokens", ctypes.c_uint32),
        ("completion_tokens", ctypes.c_uint32),
        ("rounds", ctypes.c_uint32),
        ("drafts_accepted", ctypes.c_uint32),
        ("drafts_proposed", ctypes.c_uint32),
        ("cached_tokens", ctypes.c_uint32),
        ("prefill_ms", ctypes.c_double),
        ("decode_ms", ctypes.c_double),
        ("total_ms", ctypes.c_double),
        ("tok_per_sec", ctypes.c_double),
    ]


_TOKEN_CB = ctypes.CFUNCTYPE(
    ctypes.c_uint8,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)

_WAVE_CB = ctypes.CFUNCTYPE(
    ctypes.c_uint8,
    ctypes.c_void_p,
    ctypes.c_uint32,
    ctypes.c_uint32,
    ctypes.POINTER(ctypes.c_char),
    ctypes.c_size_t,
)

_LIB: Optional[ctypes.CDLL] = None


def _load_dylib() -> ctypes.CDLL:
    global _LIB
    if _LIB is not None:
        return _LIB

    candidates = [
        os.environ.get("MLX_TF_LIB", ""),
        os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "zig-out", "lib", "libmlxtf.dylib")),
        os.path.abspath("zig-out/lib/libmlxtf.dylib"),
        os.path.abspath("tensorfold/zig-out/lib/libmlxtf.dylib"),
    ]
    lib = None
    for path in candidates:
        if path and os.path.exists(path):
            lib = ctypes.CDLL(path)
            break
    if lib is None:
        raise RuntimeError(
            "libmlxtf.dylib not found. Build TensorFold mlx_tf with `zig build --release=fast` "
            "or set MLX_TF_LIB=/path/to/libmlxtf.dylib."
        )

    lib.mlxtf_version.argtypes = []
    lib.mlxtf_version.restype = ctypes.c_char_p
    lib.mlxtf_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]
    lib.mlxtf_load.restype = ctypes.c_void_p
    lib.mlxtf_load_ex.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.c_uint8,
        ctypes.c_uint8,
    ]
    lib.mlxtf_load_ex.restype = ctypes.c_void_p
    lib.mlxtf_free.argtypes = [ctypes.c_void_p]
    lib.mlxtf_free.restype = None
    lib.mlxtf_device_name.argtypes = [ctypes.c_void_p]
    lib.mlxtf_device_name.restype = ctypes.c_char_p
    lib.mlxtf_arch_name.argtypes = [ctypes.c_void_p]
    lib.mlxtf_arch_name.restype = ctypes.c_char_p
    lib.mlxtf_uses_m5_tensor_ops.argtypes = [ctypes.c_void_p]
    lib.mlxtf_uses_m5_tensor_ops.restype = ctypes.c_uint8
    lib.mlxtf_has_vision.argtypes = [ctypes.c_void_p]
    lib.mlxtf_has_vision.restype = ctypes.c_uint8
    lib.mlxtf_encode_image.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.mlxtf_encode_image.restype = ctypes.c_int32
    lib.mlxtf_encode_patches.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_uint16),
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_uint32),
    ]
    lib.mlxtf_encode_patches.restype = ctypes.c_int32
    lib.mlxtf_generate.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(_MlxTfGenOptions),
        _TOKEN_CB,
        ctypes.c_void_p,
        ctypes.POINTER(_MlxTfStats),
    ]
    lib.mlxtf_generate.restype = ctypes.c_int32
    lib.mlxtf_generate_with_images.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.c_uint32,
        ctypes.POINTER(_MlxTfGenOptions),
        _TOKEN_CB,
        ctypes.c_void_p,
        ctypes.POINTER(_MlxTfStats),
    ]
    lib.mlxtf_generate_with_images.restype = ctypes.c_int32
    lib.mlxtf_stream_parallel.argtypes = [
        ctypes.c_void_p,
        ctypes.c_char_p,
        ctypes.c_size_t,
        ctypes.c_uint32,
        ctypes.POINTER(ctypes.c_char_p),
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.POINTER(ctypes.c_uint32),
        ctypes.POINTER(ctypes.c_float),
        _WAVE_CB,
        ctypes.c_void_p,
        ctypes.POINTER(_MlxTfStats),
    ]
    lib.mlxtf_stream_parallel.restype = ctypes.c_int32
    lib.mlxtf_serve.argtypes = [ctypes.c_void_p, ctypes.c_uint16, ctypes.c_char_p]
    lib.mlxtf_serve.restype = ctypes.c_int32
    lib.mlxtf_serve_host.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint16, ctypes.c_char_p]
    lib.mlxtf_serve_host.restype = ctypes.c_int32
    lib.mlxtf_convert.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint32, ctypes.c_uint32]
    lib.mlxtf_convert.restype = ctypes.c_int32
    lib.mlxtf_convert_tfq3.argtypes = [
        ctypes.c_char_p,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_char_p,
        ctypes.c_uint32,
        ctypes.c_uint8,
    ]
    lib.mlxtf_convert_tfq3.restype = ctypes.c_int32

    _LIB = lib
    return lib


def resolve_model_path(path_or_repo: str, allow_download: bool = True) -> str:
    """Resolve a local model directory, local HF cache snapshot, or auto-download from HuggingFace Hub."""
    expanded = os.path.expanduser(path_or_repo)
    if os.path.isdir(expanded):
        return os.path.abspath(expanded)

    # Check local HuggingFace cache first (works offline, zero network overhead)
    hub_dir = os.path.expanduser(os.environ.get("HF_HUB_CACHE", "~/.cache/huggingface/hub"))
    cache_key = "models--" + path_or_repo.strip("/").replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(hub_dir, cache_key, "snapshots", "*")))
    for snap in reversed(snaps):
        if os.path.exists(os.path.join(snap, "config.json")):
            return snap

    if allow_download and "/" in path_or_repo:
        try:
            from huggingface_hub import snapshot_download
            return snapshot_download(repo_id=path_or_repo)
        except ImportError:
            pass

    raise FileNotFoundError(
        f"Model '{path_or_repo}' not found locally or in {hub_dir}. "
        "Pass a local directory or install `huggingface_hub` for automatic downloads."
    )


@dataclass
class GenerationResponse:
    text: str
    token: int
    prompt_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tokens: int = 0
    generation_tps: float = 0.0
    finish_reason: Optional[str] = None


@dataclass
class ParallelWaveResult:
    texts: list[str]
    token_ids: list[list[int]]
    prefix_tokens: int
    cached_tokens: int
    total_generated: int
    decode_ms: float
    total_ms: float
    tok_per_sec: float


class MlxTfTokenizer:
    """Tokenizer wrapper mirroring mlx_lm's chat template interface across all 8 supported model families."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.eos_token_ids: set[int] = set()
        self.chat_style: str = "qwen"
        self.has_think_token: bool = False

        # Auto-detect exact chat template style from tokenizer.json special tokens (matching native tokenizer.zig)
        tok_p = os.path.join(model_dir, "tokenizer.json")
        if os.path.exists(tok_p):
            try:
                with open(tok_p, "r", encoding="utf-8") as f:
                    tj = json.load(f)
                for item in tj.get("added_tokens", []):
                    c = item.get("content", "")
                    if c == "<|start_header_id|>":
                        self.chat_style = "llama3"
                    elif c == "<start_of_turn>":
                        self.chat_style = "gemma"
                    elif c == "<|turn>":
                        self.chat_style = "gemma4"
                    elif c == "[INST]":
                        self.chat_style = "mistral"
                    elif c == "<\uff5cUser\uff5c>":
                        self.chat_style = "deepseek"
                    elif c == "<|user|>":
                        self.chat_style = "glm"
                    elif c == "<|startoftext|>" and self.chat_style == "qwen":
                        self.chat_style = "lfm2"
                    if c == "<think>":
                        self.has_think_token = True
            except Exception:
                pass

        cfg_p = os.path.join(model_dir, "config.json")
        if os.path.exists(cfg_p):
            with open(cfg_p, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            tc = cfg.get("text_config", cfg) if isinstance(cfg.get("text_config"), dict) else cfg
            for key in ("eos_token_id",):
                for src in (tc, cfg):
                    val = src.get(key)
                    if isinstance(val, int):
                        self.eos_token_ids.add(val)
                    elif isinstance(val, list):
                        self.eos_token_ids.update(int(x) for x in val)

    def apply_chat_template(
        self,
        messages: Sequence[dict[str, str]],
        tokenize: bool = False,
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        if tokenize:
            raise ValueError("mlx_tf passes UTF-8 strings directly to the native Zig BPE tokenizer; use tokenize=False.")
        style = self.chat_style
        parts: list[str] = []

        if style == "llama3":
            parts.append("<|begin_of_text|>")
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>")
            if add_generation_prompt:
                parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        elif style == "mistral":
            parts.append("<s>")
            sys_prefix = ""
            sys_used = False
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    sys_prefix = content
                elif role == "user":
                    body = f"{sys_prefix}\n\n{content}" if (sys_prefix and not sys_used) else content
                    if sys_prefix:
                        sys_used = True
                    parts.append(f"[INST] {body} [/INST]")
                else:
                    parts.append(f" {content}</s>")
        elif style == "gemma":
            parts.append("<bos>")
            sys_prefix = ""
            sys_used = False
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    sys_prefix = content
                    continue
                r = "model" if role == "assistant" else role
                body = f"{sys_prefix}\n\n{content}" if (sys_prefix and not sys_used and r == "user") else content
                if sys_prefix and r == "user":
                    sys_used = True
                parts.append(f"<start_of_turn>{r}\n{body}<end_of_turn>\n")
            if add_generation_prompt:
                parts.append("<start_of_turn>model\n")
        elif style == "gemma4":
            parts.append("<bos>")
            for msg in messages:
                r = "model" if msg.get("role", "user") == "assistant" else msg.get("role", "user")
                parts.append(f"<|turn>{r}\n{msg.get('content', '')}<turn|>\n")
            if add_generation_prompt:
                parts.append("<|turn>model\n")
        elif style == "deepseek":
            parts.append("<\uff5cbegin\u2581of\u2581sentence\uff5c>")
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    parts.append(f"{content}\n\n")
                elif role == "user":
                    parts.append(f"<\uff5cUser\uff5c>{content}")
                else:
                    parts.append(f"<\uff5cAssistant\uff5c>{content}<\uff5cend\u2581of\u2581sentence\uff5c>")
            if add_generation_prompt:
                parts.append("<\uff5cAssistant\uff5c>")
                if self.has_think_token:
                    parts.append("<think>\n" if enable_thinking else "<think>\n\n</think>\n\n")
        elif style == "glm":
            parts.append("[gMASK]<sop>")
            for msg in messages:
                role = msg.get("role", "user")
                tag = "<|system|>" if role == "system" else ("<|assistant|>" if role == "assistant" else "<|user|>")
                parts.append(f"{tag}\n{msg.get('content', '')}")
            if add_generation_prompt:
                parts.append("<|assistant|>\n")
        elif style == "lfm2":
            parts.append("<|startoftext|>")
            for msg in messages:
                parts.append(f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>\n")
            if add_generation_prompt:
                parts.append("<|im_start|>assistant\n")
        else:
            for msg in messages:
                parts.append(f"<|im_start|>{msg.get('role', 'user')}\n{msg.get('content', '')}<|im_end|>\n")
            if add_generation_prompt:
                if self.has_think_token:
                    parts.append("<|im_start|>assistant\n<think>\n" if enable_thinking else "<|im_start|>assistant\n<think>\n\n</think>\n\n")
                else:
                    parts.append("<|im_start|>assistant\n")
        return "".join(parts)


class MlxTfModel:
    """Handle to a bare-Metal TensorFold `mlx_tf` model loaded in `libmlxtf.dylib`."""

    def __init__(
        self,
        model_dir: str,
        draft_dir: Optional[str] = None,
        cross_draft_dir: Optional[str] = None,
        max_context: int = 8192,
        max_streams: int = 16,
        force_simd: bool = False,
        keep_bf16: bool = False,
    ):
        self._lib = _load_dylib()
        self.model_dir = model_dir
        self.draft_dir = draft_dir
        self.cross_draft_dir = cross_draft_dir
        self.max_context = max_context
        ptr = self._lib.mlxtf_load_ex(
            model_dir.encode("utf-8"),
            draft_dir.encode("utf-8") if draft_dir else None,
            cross_draft_dir.encode("utf-8") if cross_draft_dir else None,
            ctypes.c_uint32(max_context),
            ctypes.c_uint32(max_streams),
            ctypes.c_uint8(1 if force_simd else 0),
            ctypes.c_uint8(1 if keep_bf16 else 0),
        )
        if not ptr:
            raise RuntimeError(f"TensorFold mlxtf_load failed for '{model_dir}'")
        self._ptr = ctypes.c_void_p(ptr)
        self.device_name = self._lib.mlxtf_device_name(self._ptr).decode("utf-8", errors="replace")
        self.arch_name = self._lib.mlxtf_arch_name(self._ptr).decode("utf-8", errors="replace")
        self.uses_m5_tensor_ops = bool(self._lib.mlxtf_uses_m5_tensor_ops(self._ptr))
        self.has_vision = bool(self._lib.mlxtf_has_vision(self._ptr))

    def __enter__(self) -> "MlxTfModel":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def close(self) -> None:
        if getattr(self, "_ptr", None):
            self._lib.mlxtf_free(self._ptr)
            self._ptr = None

    def __del__(self) -> None:
        self.close()

    def encode_image(self, image_path_or_uri: str, max_elems: int = 4096 * 4096) -> tuple[list[float], int, int]:
        """Encode an image (path, file://, or data: URI) with the Metal vision tower -> (flat_f32, num_tokens, hidden_size)."""
        if not self.has_vision:
            raise RuntimeError(f"Model '{self.model_dir}' has no vision encoder.")
        buf = (ctypes.c_float * max_elems)()
        nt, hd = ctypes.c_uint32(0), ctypes.c_uint32(0)
        rc = self._lib.mlxtf_encode_image(
            self._ptr, image_path_or_uri.encode("utf-8"), buf, max_elems, ctypes.byref(nt), ctypes.byref(hd)
        )
        if rc != 0:
            raise RuntimeError(f"mlxtf_encode_image failed with rc={rc} for '{image_path_or_uri}'")
        total = int(nt.value) * int(hd.value)
        return list(buf[:total]), int(nt.value), int(hd.value)

    def serve(self, port: int = 8080, host: str = "127.0.0.1", model_name: Optional[str] = None) -> None:
        name_b = (model_name or os.path.basename(self.model_dir)).encode("utf-8")
        rc = self._lib.mlxtf_serve_host(self._ptr, host.encode("utf-8"), ctypes.c_uint16(port), name_b)
        if rc != 0:
            raise RuntimeError(f"mlxtf_serve_host exited with code {rc}")


def load(
    path_or_hf_repo: str,
    draft_model: Optional[str] = None,
    cross_draft_model: Optional[str] = None,
    max_context: int = 8192,
    max_streams: int = 16,
    force_simd: bool = False,
    keep_bf16: bool = False,
    allow_download: bool = True,
) -> tuple[MlxTfModel, MlxTfTokenizer]:
    """Drop-in replacement for `mlx_lm.load(...)` with zero Apple MLX dependency."""
    mdir = resolve_model_path(path_or_hf_repo, allow_download=allow_download)
    ddir = resolve_model_path(draft_model, allow_download=allow_download) if draft_model else None
    cdir = resolve_model_path(cross_draft_model, allow_download=allow_download) if cross_draft_model else None
    model = MlxTfModel(
        mdir, ddir, cdir, max_context=max_context, max_streams=max_streams, force_simd=force_simd, keep_bf16=keep_bf16
    )
    return model, MlxTfTokenizer(mdir)


def stream_generate(
    model: MlxTfModel,
    tokenizer: Optional[MlxTfTokenizer] = None,
    prompt: str = "",
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int = 42,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[Sequence[str]] = None,
    images: Optional[Sequence[str]] = None,
    use_chat_template: Optional[bool] = None,
    enable_thinking: bool = False,
) -> Iterator[GenerationResponse]:
    """Drop-in streaming generator mirroring `mlx_lm.stream_generate`, with stop-string & VLM image support."""
    del tokenizer
    stop_seqs = [s for s in (stop or []) if s]
    max_stop_len = max((len(s) for s in stop_seqs), default=0)
    img_list = list(images) if images else []
    if use_chat_template is None:
        chat_flag = 1 if (img_list and "<|im_start|>" not in prompt and "<start_of_turn>" not in prompt) else 0
    else:
        chat_flag = 1 if use_chat_template else 0

    q: queue.Queue[Optional[tuple[int, str] | Exception]] = queue.Queue()
    stats = _MlxTfStats()
    opts = _MlxTfGenOptions(
        max_tokens=max_tokens,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        min_p=min_p,
        seed=seed,
        enable_thinking=1 if enable_thinking else 0,
        use_chat_template=chat_flag,
        stop_on_eos=1,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
    )
    cancelled = threading.Event()
    pbytes = prompt.encode("utf-8")
    img_bytes = [p.encode("utf-8") for p in img_list]
    img_arr = (ctypes.c_char_p * len(img_bytes))(*img_bytes) if img_bytes else None

    @_TOKEN_CB
    def _on_token(_ud, tok_id, utf8_ptr, utf8_len):
        if cancelled.is_set():
            return 0
        piece = ctypes.string_at(utf8_ptr, utf8_len).decode("utf-8", errors="replace")
        q.put((int(tok_id), piece))
        return 1

    def _worker():
        try:
            if img_arr is not None:
                rc = model._lib.mlxtf_generate_with_images(
                    model._ptr, pbytes, len(pbytes), img_arr, ctypes.c_uint32(len(img_bytes)),
                    ctypes.byref(opts), _on_token, None, ctypes.byref(stats),
                )
            else:
                rc = model._lib.mlxtf_generate(
                    model._ptr, pbytes, len(pbytes), ctypes.byref(opts), _on_token, None, ctypes.byref(stats)
                )
            if rc == -5:
                q.put(ValueError("Prompt encoded to 0 tokens."))
            elif rc == -6:
                q.put(ValueError(f"Prompt exceeds max_context ({model.max_context}). Increase max_context in mlx_tf.load()."))
            elif rc == -10:
                q.put(RuntimeError(f"Model '{model.model_dir}' does not include a vision encoder."))
            elif rc != 0:
                q.put(RuntimeError(f"mlxtf_generate failed with rc={rc}"))
        except Exception as exc:
            q.put(exc)
        finally:
            q.put(None)

    t0 = time.perf_counter()
    th = threading.Thread(target=_worker, daemon=True)
    th.start()

    n_gen = 0
    last_tok = 0
    t_first_tok: Optional[float] = None
    pending = ""
    try:
        while True:
            item = q.get()
            if item is None:
                th.join(timeout=5.0)
                p_tps = (stats.prompt_tokens / (stats.prefill_ms / 1000.0)) if stats.prefill_ms > 0 else 0.0
                g_tps = stats.tok_per_sec if stats.tok_per_sec > 0 else (
                    ((n_gen - 1) / max(time.perf_counter() - t_first_tok, 1e-6)) if (t_first_tok and n_gen > 1) else 0.0
                )
                yield GenerationResponse(
                    text=pending,
                    token=last_tok,
                    prompt_tokens=stats.prompt_tokens,
                    prompt_tps=p_tps,
                    generation_tokens=n_gen,
                    generation_tps=g_tps,
                    finish_reason="length" if n_gen >= max_tokens else "stop",
                )
                break
            if isinstance(item, Exception):
                raise item
            tok_id, piece = item
            n_gen += 1
            last_tok = tok_id
            now = time.perf_counter()
            if t_first_tok is None:
                t_first_tok = now
                cur_tps = 0.0
            else:
                cur_tps = (n_gen - 1) / max(now - t_first_tok, 1e-6)

            if not stop_seqs:
                yield GenerationResponse(
                    text=piece, token=tok_id, prompt_tokens=stats.prompt_tokens,
                    generation_tokens=n_gen, generation_tps=cur_tps,
                )
            else:
                pending += piece
                hit_pos = None
                for s in stop_seqs:
                    idx = pending.find(s)
                    if idx != -1 and (hit_pos is None or idx < hit_pos):
                        hit_pos = idx
                if hit_pos is not None:
                    emit = pending[:hit_pos]
                    pending = ""
                    cancelled.set()
                    if emit:
                        yield GenerationResponse(
                            text=emit, token=tok_id, prompt_tokens=stats.prompt_tokens,
                            generation_tokens=n_gen, generation_tps=cur_tps,
                            finish_reason="stop",
                        )
                    break
                safe_len = len(pending) - (max_stop_len - 1)
                if safe_len > 0:
                    emit, pending = pending[:safe_len], pending[safe_len:]
                    yield GenerationResponse(
                        text=emit, token=tok_id, prompt_tokens=stats.prompt_tokens,
                        generation_tokens=n_gen, generation_tps=cur_tps,
                    )
    finally:
        cancelled.set()
        th.join(timeout=5.0)


def generate(
    model: MlxTfModel,
    tokenizer: Optional[MlxTfTokenizer] = None,
    prompt: str = "",
    max_tokens: int = 256,
    temp: float = 0.0,
    top_p: float = 1.0,
    top_k: int = 0,
    min_p: float = 0.0,
    seed: int = 42,
    repetition_penalty: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    stop: Optional[Sequence[str]] = None,
    images: Optional[Sequence[str]] = None,
    use_chat_template: Optional[bool] = None,
    enable_thinking: bool = False,
    verbose: bool = False,
) -> str:
    """Drop-in replacement for `mlx_lm.generate(...)` with optional multimodal `images=` support."""
    t0 = time.perf_counter()
    chunks: list[str] = []
    n_tok = 0
    for resp in stream_generate(
        model, tokenizer, prompt=prompt, max_tokens=max_tokens,
        temp=temp, top_p=top_p, top_k=top_k, min_p=min_p, seed=seed,
        repetition_penalty=repetition_penalty,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        stop=stop,
        images=images,
        use_chat_template=use_chat_template,
        enable_thinking=enable_thinking,
    ):
        chunks.append(resp.text)
        n_tok += 1
        if verbose:
            print(resp.text, end="", flush=True)
    text = "".join(chunks)
    if verbose:
        dt = max(time.perf_counter() - t0, 1e-6)
        print(f"\n[mlx_tf · TensorFold] {n_tok} tokens in {dt:.2f}s ({n_tok / dt:.1f} tok/s) on {model.device_name} [{model.arch_name}]")
    return text


def stream_parallel(
    model: MlxTfModel,
    shared_prompt: str,
    delta_prompts: Sequence[str],
    max_tokens: int | Sequence[int] = 128,
    temp: float | Sequence[float] = 0.0,
) -> ParallelWaveResult:
    """Zero-copy prefix-forked lockstep multi-stream wave (up to 16 streams in 1 tensor tile)."""
    n = len(delta_prompts)
    if n == 0:
        return ParallelWaveResult([], [], 0, 0, 0, 0.0, 0.0, 0.0)
    shared_b = shared_prompt.encode("utf-8")
    deltas_b = [d.encode("utf-8") for d in delta_prompts]
    d_ptrs = (ctypes.c_char_p * n)(*deltas_b)
    d_lens = (ctypes.c_size_t * n)(*[len(b) for b in deltas_b])
    mt_list = [max_tokens] * n if isinstance(max_tokens, int) else list(max_tokens)
    tp_list = [temp] * n if isinstance(temp, (int, float)) else list(temp)
    mt_arr = (ctypes.c_uint32 * n)(*mt_list)
    tp_arr = (ctypes.c_float * n)(*tp_list)

    texts: list[list[str]] = [[] for _ in range(n)]
    toks: list[list[int]] = [[] for _ in range(n)]
    stats = _MlxTfStats()

    @_WAVE_CB
    def _on_wave(_ud, sidx, tok_id, utf8_ptr, utf8_len):
        texts[sidx].append(ctypes.string_at(utf8_ptr, utf8_len).decode("utf-8", errors="replace"))
        toks[sidx].append(int(tok_id))
        return 1

    rc = model._lib.mlxtf_stream_parallel(
        model._ptr, shared_b, len(shared_b), ctypes.c_uint32(n),
        d_ptrs, d_lens, mt_arr, tp_arr, _on_wave, None, ctypes.byref(stats),
    )
    if rc != 0:
        raise RuntimeError(f"mlxtf_stream_parallel failed with rc={rc}")
    return ParallelWaveResult(
        texts=["".join(t) for t in texts],
        token_ids=toks,
        prefix_tokens=stats.prompt_tokens,
        cached_tokens=stats.cached_tokens,
        total_generated=stats.completion_tokens,
        decode_ms=stats.decode_ms,
        total_ms=stats.total_ms,
        tok_per_sec=stats.tok_per_sec,
    )


def convert(
    hf_path: str,
    mlx_path: str = "mlx_model",
    q_bits: int = 4,
    q_group_size: int = 64,
    bits: Optional[int] = None,
    group_size: Optional[int] = None,
    format: str = "affine",
    profile: str = "",
    head_bits: int = 0,
    quantize_head: bool = True,
) -> str:
    """Quantize a raw HuggingFace BF16/FP16/F32 checkpoint to MLX affine (2..8-bit) or Native TFQ3 (3/3.5/4/5-bit) on disk (zero Apple MLX)."""
    lib = _load_dylib()
    src = resolve_model_path(hf_path)
    dst = os.path.abspath(os.path.expanduser(mlx_path))
    b = bits if bits is not None else q_bits
    if format == "tfq3":
        rc = lib.mlxtf_convert_tfq3(
            src.encode("utf-8"),
            dst.encode("utf-8"),
            ctypes.c_uint32(b),
            profile.encode("utf-8") if profile else None,
            ctypes.c_uint32(head_bits),
            ctypes.c_uint8(1 if quantize_head else 0),
        )
        if rc != 0:
            raise RuntimeError(f"mlxtf_convert_tfq3 failed with rc={rc} for {src} -> {dst}")
        return dst
    gs = group_size if group_size is not None else q_group_size
    rc = lib.mlxtf_convert(src.encode("utf-8"), dst.encode("utf-8"), ctypes.c_uint32(b), ctypes.c_uint32(gs))
    if rc != 0:
        raise RuntimeError(f"mlxtf_convert failed with rc={rc} for {src} -> {dst}")
    return dst


__all__ = [
    "__version__",
    "MlxTfModel",
    "MlxTfTokenizer",
    "GenerationResponse",
    "ParallelWaveResult",
    "resolve_model_path",
    "load",
    "generate",
    "stream_generate",
    "stream_parallel",
    "convert",
]
