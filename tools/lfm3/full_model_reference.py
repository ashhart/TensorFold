#!/usr/bin/env python3
"""Independent FP32 decoder or actual installed official MLX BF16 decoder, propagated inputs."""
import argparse, hashlib, importlib, importlib.metadata, inspect, json, math, os, struct
from pathlib import Path
import numpy as np
from fp32_graph_reference import Graph
from fp64_reference import Oracle


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


class Checkpoint(Oracle):
    def __init__(self, directory):
        directory = Path(directory).resolve()
        self.directory = directory
        self.config = json.loads((directory / 'config.json').read_text())
        self.text = self.config.get('text_config', self.config)
        if self.text.get('model_type') != 'lfm3' or not self.text.get('tie_word_embeddings'):
            raise ValueError('Expected tied LFM3')
        index = directory / 'model.safetensors.index.json'
        self.map = {}
        self.files = {}
        declared = json.loads(index.read_text())['weight_map'] if index.exists() else None
        names = (
            sorted(set(declared.values()))
            if declared is not None
            else sorted(p.name for p in directory.glob('*.safetensors'))
        )
        if not names:
            raise ValueError('No checkpoint shards')
        for name in names:
            p = (directory / name).resolve()
            if not p.is_relative_to(directory):
                raise ValueError('Escaping checkpoint path')
            with p.open('rb') as f:
                length = struct.unpack('<Q', f.read(8))[0]
                if length > 64 << 20:
                    raise ValueError('Unbounded header')
                header = json.loads(f.read(length))
            self.files[name] = (p, 8 + length, header)
            for key in header:
                if key == '__metadata__':
                    continue
                if key in self.map:
                    raise ValueError('Duplicate tensor')
                self.map[key] = name
        if declared is not None and self.map != declared:
            raise ValueError('Index/header mismatch')
        prefixes = [p for p in ['model.', 'language_model.model.'] if p + 'embed_tokens.weight' in self.map]
        if len(prefixes) != 1:
            raise ValueError('Ambiguous decoder namespace')
        self.prefix = prefixes[0]

    def pins(self):
        paths = [self.directory / 'config.json'] + [p for p, _, _ in self.files.values()]
        index = self.directory / 'model.safetensors.index.json'
        if index.exists():
            paths.append(index)
        return {str(p): sha(p) for p in paths}


class FP32(Graph):
    def __init__(self, directory, query_tile=128):
        self.checkpoint = Checkpoint(directory)
        self.config = self.checkpoint.config
        self.text = self.checkpoint.text
        self.prefix = self.checkpoint.prefix
        self.boundaries = 'fp32'
        self.trace = None
        self.query_tile = query_tile

    def attention_value(self, q, k, v, window=0, causal=True):
        # Query tiling bounds the score allocation. Each query still uses the full original K reduction.
        out = np.empty_like(q)
        key_positions = np.arange(k.shape[1])[None, :]
        for start in range(0, q.shape[1], self.query_tile):
            stop = min(start + self.query_tile, q.shape[1])
            scores = (q[:, start:stop] @ k.swapaxes(-1, -2)) / np.float32(math.sqrt(q.shape[-1]))
            if causal:
                distance = np.arange(start, stop)[:, None] - key_positions
                allowed = distance >= 0
                if window:
                    allowed &= distance < window
                scores = np.where(allowed, scores, np.float32(-np.inf))
            p = np.exp(scores - scores.max(axis=-1, keepdims=True))
            p = p / p.sum(axis=-1, keepdims=True, dtype=np.float32)
            out[:, start:stop] = p @ v
        return out

    def selected(self, tokens, indices, image_rows=None, image_start=None):
        hidden = self.weights(self.prefix + 'embed_tokens', tokens)
        if image_rows is not None:
            inject_check(tokens, image_rows, image_start, hidden.shape[-1])
            hidden[image_start : image_start + len(image_rows)] = image_rows
        elif image_start is not None:
            raise ValueError('Image start without rows')
        for index, kind in enumerate(self.text['layer_types']):
            prefix = self.prefix + f'layers.{index}.'
            x = self.norm(hidden, prefix + 'operator_norm')
            result = self.conv(x, prefix) if kind == 'conv' else self.attention(x, prefix, kind)
            hidden = hidden + (
                self.norm(result, prefix + 'post_operator_norm')
                if self.text.get('use_sandwich_norm', True)
                else result
            )
            x = self.norm(hidden, prefix + 'ffn_norm')
            gate = self.linear(x, prefix + 'feed_forward.w1')
            up = self.linear(x, prefix + 'feed_forward.w3')
            result = self.linear(self.swiglu(gate, up), prefix + 'feed_forward.w2')
            hidden = hidden + (
                self.norm(result, prefix + 'post_ffn_norm')
                if self.text.get('use_sandwich_norm', True)
                else result
            )
            if not np.isfinite(hidden).all():
                raise ValueError('Nonfinite FP32 hidden')
            print('FP32 layer', index, flush=True)
        return self.linear(
            self.norm(hidden[indices], self.prefix + 'embedding_norm'), self.prefix + 'embed_tokens'
        )


def inject_check(tokens, rows, start, width):
    if (
        start is None
        or start < 0
        or rows.ndim != 2
        or rows.shape[1] != width
        or len(rows) == 0
        or start + len(rows) > len(tokens)
    ):
        raise ValueError('Invalid image rows')
    expected = np.zeros(len(tokens), bool)
    expected[start : start + len(rows)] = True
    if not np.array_equal(tokens == 125030, expected) or not np.isfinite(rows).all():
        raise ValueError('Invalid image placeholder span')


def official(a, tokens, indices, image_rows):
    import mlx.core as mx
    import mlx_vlm.utils as utils

    if importlib.metadata.version('mlx') != a.expect_mlx:
        raise ValueError('Wrong MLX version')
    if a.official_pins:
        expected = json.loads(a.official_pins.read_text())
        site = Path(utils.__file__).resolve().parent.parent
        for rel, h in expected.items():
            path = (site / rel).resolve()
            if not path.is_relative_to(site) or sha(path) != h:
                raise ValueError(('Official source pin mismatch', rel))
    model = utils.load_model(a.model_dir, lazy=False)
    lm = model.language_model
    source_paths = {
        str(Path(inspect.getfile(type(x))).resolve())
        for x in [model, lm, lm.model, lm.model.layers[0], lm.model.embed_tokens]
    }
    source_paths.add(str(Path(utils.__file__).resolve()))
    for layer in lm.model.layers:
        source_paths.add(
            str(
                Path(
                    inspect.getfile(type(layer.self_attn if layer.is_attention_layer else layer.conv))
                ).resolve()
            )
        )
    modules = ['mlx_vlm.models.base', 'mlx_vlm.models.cache']
    for name in modules:
        source_paths.add(str(Path(importlib.import_module(name).__file__).resolve()))
    sources = {p: sha(p) for p in sorted(source_paths)}
    cache = lm.make_cache()
    answers = {}
    ends = (
        list(range(a.chunk, a.prompt_len, a.chunk))
        + [a.prompt_len]
        + list(range(a.prompt_len + 1, len(tokens) + 1))
    )
    start = 0
    for end in ends:
        ids = mx.array(tokens[None, start:end], dtype=mx.int32)
        emb = None
        if image_rows is not None:
            emb = lm.model.embed_tokens(ids)
            lo = max(start, a.image_start)
            hi = min(end, a.image_start + len(image_rows))
            if lo < hi:
                emb[:, lo - start : hi - start] = mx.array(
                    image_rows[lo - a.image_start : hi - a.image_start]
                ).astype(mx.bfloat16)[None]
        output = lm(ids, cache=cache, inputs_embeds=emb)
        pick = [int(i) for i in indices if start <= i < end]
        if pick:
            y = output.logits[:, [i - start for i in pick]]
            mx.eval(y)
            if y.dtype != mx.bfloat16:
                raise ValueError(('Official logits not BF16', y.dtype))
            values = np.asarray(y.astype(mx.float32))[0]
            for i, row in zip(pick, values):
                answers[i] = row
        mx.eval([c.state for c in cache])
        start = end
    return np.stack([answers[int(i)] for i in indices]), sources


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['fp32', 'mlx'], required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt-len', type=int, required=True)
    p.add_argument(
        '--rows',
        help='Comma separated absolute token positions; default last prompt plus every teacher-forced continuation position',
    )
    p.add_argument('--chunk', type=int, default=2048)
    p.add_argument('--query-tile', type=int, default=128)
    p.add_argument('--expect-mlx', default='0.32.4')
    p.add_argument('--official-pins', type=Path)
    p.add_argument('--image-rows', type=Path)
    p.add_argument('--image-start', type=int)
    a = p.parse_args()
    tokens = np.load(a.tokens, allow_pickle=False)
    if (
        tokens.ndim != 1
        or not np.issubdtype(tokens.dtype, np.integer)
        or not 0 < a.prompt_len <= len(tokens)
        or a.chunk <= 0
        or a.query_tile <= 0
    ):
        raise ValueError('Invalid sequence geometry')
    indices = np.array(
        [int(i) for i in a.rows.split(',')] if a.rows else list(range(a.prompt_len - 1, len(tokens))),
        dtype=np.int64,
    )
    if (
        len(indices) == 0
        or len(np.unique(indices)) != len(indices)
        or indices.min() < 0
        or indices.max() >= len(tokens)
    ):
        raise ValueError('Invalid selected rows')
    ck = Checkpoint(a.model_dir)
    if tokens.min() < 0 or tokens.max() >= ck.text['vocab_size']:
        raise ValueError('Invalid token ID')
    rows = np.load(a.image_rows, allow_pickle=False).astype(np.float32) if a.image_rows else None
    if rows is not None:
        inject_check(tokens, rows, a.image_start, ck.text['hidden_size'])
        if np.any(rows.view(np.uint32) & 65535):
            raise ValueError('Fixed image rows must be identical BF16 values in both references')
    elif a.image_start is not None:
        raise ValueError('Image start requires rows')
    pins = ck.pins()
    pins[str(a.tokens)] = sha(a.tokens)
    if a.image_rows:
        pins[str(a.image_rows)] = sha(a.image_rows)
    y, sources = (
        (FP32(a.model_dir, a.query_tile).selected(tokens, indices, rows, a.image_start), {})
        if a.backend == 'fp32'
        else official(a, tokens, indices, rows)
    )
    if y.dtype != np.float32 or not np.isfinite(y).all():
        raise ValueError('Invalid captured logits')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(a.output, y)
    meta = {
        'operator': 'full_model_logits',
        'backend': a.backend,
        'rows': len(indices),
        'shape': list(y.shape),
        'selected_token_positions': indices.tolist(),
        'prompt_len': a.prompt_len,
        'teacher_forced_continuation': len(tokens) - a.prompt_len,
        'reference_chunk': a.chunk,
        'image_start': a.image_start,
        'inputs_sha256': pins,
        'output_sha256': sha(a.output),
        'source_sha256': sha(__file__),
        'support_sources_sha256': {
            str(Path(__file__).parent / n): sha(Path(__file__).parent / n)
            for n in ['fp32_graph_reference.py', 'fp64_reference.py']
        },
        'official_sources_sha256': sources,
        'numpy': np.__version__,
        'thread_environment': {
            k: os.environ.get(k)
            for k in ['VECLIB_MAXIMUM_THREADS', 'OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']
        },
        'query_tile': a.query_tile,
        'official_pins_sha256': sha(a.official_pins) if a.official_pins else None,
        'mlx_version': importlib.metadata.version('mlx') if a.backend == 'mlx' else None,
        'arithmetic': (
            'Independent CPU FP32 propagated decoder, no BF16 intermediate rounding'
            if a.backend == 'fp32'
            else 'Installed official MLX-VLM propagated BF16 decoder; official cache; prompt chunks then one token teacher-forced continuation'
        ),
        'acceptance': None,
        'scope': 'Selected full-vocabulary logits after propagated decoder; fixed projected image input only if supplied; no independent vision claim',
    }
    a.output.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')


if __name__ == '__main__':
    main()
