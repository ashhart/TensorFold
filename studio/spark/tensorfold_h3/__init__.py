"""ComfyUI node: FastH3's transformer blocks on TensorFold's CUDA engine (int8 on the tensor cores)."""
# A LoRA's block projections are merged here and requantized; the int8 copy saved beside the checkpoint never changes.

import ctypes
import logging
import math
import os
import time
import uuid

import torch

import comfy.patcher_extension
import folder_paths
from comfy.ldm.minimax.model import MiniMaxH3Model

TILE = 64
CUBE = (4, 4, 4)


class Inputs(ctypes.Structure):
    _fields_ = [("x", ctypes.c_uint64), ("tables", ctypes.c_uint64), ("line", ctypes.c_uint64), ("cos", ctypes.c_uint64),
                ("sin", ctypes.c_uint64), ("slot", ctypes.c_uint64), ("sizes", ctypes.c_uint64), ("rows", ctypes.c_uint32),
                ("lines", ctypes.c_uint32), ("rot", ctypes.c_uint32), ("tiles", ctypes.c_uint32),
                ("prefix_tiles", ctypes.c_uint32), ("keep", ctypes.c_uint32), ("blocks", ctypes.c_uint32),
                ("attention_out", ctypes.c_uint64)]


_library = None
_engines = {}   # checkpoint path -> engine handle; the int8 weights stay on the GPU while ComfyUI runs
_merged = {}    # checkpoint path -> {model key: the patches merged into that projection}

# the engine's projections, by their names in the model and the number tf_h3_set_weight knows them by
PROJECTIONS = {"attn.qkv_proj.weight": 0, "attn.to_gate_compress.weight": 1, "attn.out_proj.weight": 2,
               "mlp.fc1.weight": 3, "mlp.fc2.weight": 4}
# what else of a block the engine holds: a patch on it cannot be honoured
HELD = ("norm1.", "norm2.", "attn.q_norm.", "attn.k_norm.", "attn.qkv_proj.", "attn.to_gate_compress.",
        "attn.out_proj.", "mlp.fc1.", "mlp.fc2.")


def library():
    global _library
    if _library is None:
        path = os.environ.get("TF_H3_LIB") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "libtf_h3.so")
        lib = ctypes.CDLL(path)
        lib.tf_h3_open.restype = ctypes.c_void_p
        lib.tf_h3_open.argtypes = [ctypes.c_char_p]
        lib.tf_h3_close.argtypes = [ctypes.c_void_p]
        lib.tf_h3_shape.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32 * 7)]
        lib.tf_h3_forward.restype = ctypes.c_double
        lib.tf_h3_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(Inputs)]
        lib.tf_h3_profile.argtypes = [ctypes.c_void_p, ctypes.c_int]
        lib.tf_h3_spent.restype = ctypes.c_int
        lib.tf_h3_spent.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_double * 16), ctypes.POINTER(ctypes.c_char_p * 16)]
        if hasattr(lib, "tf_h3_set_weight"):
            lib.tf_h3_set_weight.restype = ctypes.c_int
            lib.tf_h3_set_weight.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint64,
                                             ctypes.c_uint32, ctypes.c_uint32]
        _library = lib
    return _library


def engine(path):
    handle = _engines.get(path)
    if handle is None:
        torch.cuda.synchronize()
        handle = library().tf_h3_open(path.encode())
        if not handle:
            raise RuntimeError(f"TensorFold H3: could not load {path}")
        shape = (ctypes.c_uint32 * 7)()
        library().tf_h3_shape(handle, ctypes.byref(shape))
        logging.info(f"TensorFold H3: {shape[4]} blocks of {shape[0]}/{shape[1]}/{shape[3]} in int8, "
                     f"{'read from the saved copy' if shape[6] else 'converted from the checkpoint (saved for the next start)'} in {shape[5] / 1000:.1f} s")
        _engines[path] = handle
    return handle


def release():
    for handle in _engines.values():
        library().tf_h3_close(handle)
    _engines.clear()
    _merged.clear()


def take_block_patches(patcher):
    """The patches on the engine's projections, removed from `patcher` (a clone): {model key: patches}."""
    taken = {}
    for key in list(patcher.patches):
        if not (isinstance(key, str) and key.startswith("diffusion_model.blocks.")):
            continue
        part = key.split(".", 3)[3]                     # diffusion_model.blocks.N.<part>
        if part in PROJECTIONS:
            taken[key] = patcher.patches.pop(key)
        elif part.startswith(HELD):
            raise ValueError(f"TensorFold H3: a LoRA patches {key}, which the engine holds and cannot take; "
                             "render this LoRA with --engine comfy")
    return taken


def merge(handle, path, patches):
    """Bring the engine's projections to the checkpoint's weights with `patches` merged, redoing only changed ones."""
    import comfy.float
    import comfy.lora
    import comfy.model_management
    import comfy.utils
    from safetensors import safe_open

    done = _merged.setdefault(path, {})
    def same(a, b):
        return a is not None and b is not None and len(a) == len(b) and all(
            x[1] is y[1] and x[0] == y[0] and x[2] == y[2] and x[3] == y[3] for x, y in zip(a, b))
    todo = sorted(k for k in set(patches) | set(done) if not same(patches.get(k), done.get(k)))
    if not todo:
        return
    if not hasattr(library(), "tf_h3_set_weight"):
        raise RuntimeError("TensorFold H3: this engine cannot take a LoRA; rebuild it (spark/build-engine.sh)")
    mark = time.perf_counter()
    device = torch.device("cuda")
    compute = comfy.model_management.lora_compute_dtype(device)
    torch.cuda.synchronize()
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in todo:
            block, part = key.split(".", 3)[2:]
            weight = f.get_tensor(key[len("diffusion_model."):])
            if key in patches:
                w = comfy.lora.calculate_weight(patches[key], weight.to(device, dtype=compute, copy=True), key)
                w = comfy.float.stochastic_rounding(w, torch.bfloat16, seed=comfy.utils.string_to_seed(key))
            else:
                w = weight.to(device)
            w = w.to(torch.bfloat16).contiguous()
            torch.cuda.synchronize()
            if library().tf_h3_set_weight(handle, int(block), PROJECTIONS[part], w.data_ptr(), w.shape[0], w.shape[1]):
                done.clear()   # the engine is part merged: the next call redoes everything
                raise RuntimeError(f"TensorFold H3: the engine refused {key} {tuple(w.shape)}")
            del w, weight
            if key in patches:
                done[key] = list(patches[key])
            else:
                done.pop(key, None)
    try:
        # the checkpoint's pages the merge read are not needed again: give them back
        fd = os.open(path, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)
    except OSError:
        pass
    merged = sum(1 for k in todo if k in patches)
    logging.info(f"TensorFold H3: LoRA merged into {merged} projections"
                 + (f", {len(todo) - merged} put back to the checkpoint's" if len(todo) > merged else "")
                 + f", in {time.perf_counter() - mark:.1f} s")


def tile_plan(layout, device):
    """FastVideo's tile order: prefix in 64-row tiles, video in 4x4x4 cubes; (row slots, tile rows, prefix tiles)."""
    _text, latent_t, latent_h, latent_w, _audio = layout.signature
    grid = (int(latent_t), int(latent_h) // 2, int(latent_w) // 2)
    tiles, prefix = [], 0
    for a, b, kind in layout.segments:
        n = b - a
        if kind != "video":
            m = (n + TILE - 1) // TILE
            seg = torch.full((m * TILE,), -1, dtype=torch.int64)
            seg[:n] = torch.arange(a, b)
            tiles.append(seg.view(m, TILE))
            prefix += m
            continue
        if grid[0] * grid[1] * grid[2] != n:
            raise RuntimeError(f"TensorFold H3: video segment of {n} rows does not match the latent grid {grid}")
        pt, ph, pw = ((g + c - 1) // c * c for g, c in zip(grid, CUBE))
        padded = torch.full((pt, ph, pw), -1, dtype=torch.int64)
        padded[:grid[0], :grid[1], :grid[2]] = torch.arange(a, b).view(*grid)
        cubes = (padded.view(pt // CUBE[0], CUBE[0], ph // CUBE[1], CUBE[1], pw // CUBE[2], CUBE[2])
                 .permute(0, 2, 4, 1, 3, 5).reshape(-1, TILE))
        order = torch.argsort((cubes < 0).to(torch.int8), dim=1, stable=True)
        tiles.append(torch.gather(cubes, 1, order))
    tiles = torch.cat(tiles)
    src = tiles.reshape(-1)
    live = src >= 0
    slot = torch.empty(layout.seq_len, dtype=torch.int64)
    slot[src[live]] = torch.nonzero(live).flatten()
    return (slot.to(torch.int32).to(device), (tiles >= 0).sum(1).to(torch.int32).to(device), prefix)


def cosine(a, b):
    a, b = a.float().flatten(), b.float().flatten()
    return float(torch.dot(a, b) / (a.norm() * b.norm()))


class Blocks:
    """One patched model's state: the engine, and what stays the same through a sampling run."""

    def __init__(self, diffusion_model, path, sparsity, patches=None):
        self.model = diffusion_model
        self.path = path
        self.sparsity = sparsity
        self.patches = patches or {}
        self.plans = {}
        self.checked = False
        self.passes = 0

    def plan(self, layout, device):
        key = (tuple(layout.signature), tuple(layout.segments))
        if key not in self.plans:
            self.plans = {key: tile_plan(layout, device)}
        return self.plans[key]

    def run(self, args, blocks=0, dense=False):
        h, layout = args["img"], args["layout"]
        device = h.device
        rows = h.shape[0]
        if not h.is_contiguous() or h.dtype != torch.bfloat16:
            raise RuntimeError(f"TensorFold H3 wants a contiguous bf16 stream, got {h.dtype}")
        tables = torch.stack([torch.stack(block.adaln_proj(args["t_emb"]), dim=1) for block in self.model.blocks])
        tables = tables.to(torch.float32).contiguous()                      # (blocks, lines, 6, hidden)
        line = torch.empty(rows, dtype=torch.int32, device=device)
        for a, b, row in args["mod_segments"]:
            line[a:b] = row if isinstance(row, int) else row.to(torch.int32)
        rope = args["rope_freqs"]                                            # (1, rows, 1, rot, 2, 2): c, -s, s, c
        cos = rope[0, :, 0, :, 0, 0].to(torch.float32).contiguous()
        sin = rope[0, :, 0, :, 1, 0].to(torch.float32).contiguous()
        slot, sizes, prefix = self.plan(layout, device)
        video_tiles = sizes.shape[0] - prefix
        keep = 0 if dense or self.sparsity <= 0 else max(1, min(math.ceil((1.0 - self.sparsity) * video_tiles), video_tiles))
        inputs = Inputs(x=h.data_ptr(), tables=tables.data_ptr(), line=line.data_ptr(), cos=cos.data_ptr(), sin=sin.data_ptr(),
                        slot=slot.data_ptr(), sizes=sizes.data_ptr(), rows=rows, lines=tables.shape[1], rot=cos.shape[1],
                        tiles=sizes.shape[0], prefix_tiles=prefix, keep=keep, blocks=blocks, attention_out=0)
        handle = engine(self.path)
        merge(handle, self.path, self.patches)
        # the engine's kernels go on the default stream: whatever torch has queued on its own must have landed
        torch.cuda.synchronize()
        took = library().tf_h3_forward(handle, ctypes.byref(inputs))
        if took < 0:
            raise RuntimeError("TensorFold H3: the forward failed")
        return took, keep, sizes.shape[0]

    def check(self, args):
        """The engine against ComfyUI's own blocks on this pass's stream: after one block and after all of them."""
        if self.patches:
            logging.info("TensorFold H3 check: ComfyUI's blocks run without the LoRA here (the engine holds it), "
                         "so the figures below include the LoRA's own effect")
        start = args["img"].clone()
        options = args["transformer_options"]

        def reference(count):
            h = start.clone()
            for block in self.model.blocks[:count]:
                h = block(h, args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=options)
            return h

        for count in (1, len(self.model.blocks)):
            want = reference(count)
            for label, dense in (("dense", True), ("sparse", False)):
                if not dense and self.sparsity <= 0:
                    continue
                got = start.clone()
                self.run({**args, "img": got}, blocks=count, dense=dense)
                error = float((got.float() - want.float()).norm() / want.float().norm())
                logging.info(f"TensorFold H3 check: {count} block(s), engine {label} against ComfyUI dense: "
                             f"cosine {cosine(got, want):.5f}, relative error {error:.4f}")
            del want

    def sources(self, args):
        """TF_H3_CHECK=2: the cost of each int8 rounding, emulated one at a time in ComfyUI's own block."""
        import comfy.ops

        options = args["transformer_options"]

        def rows(x, group=0):
            # int8 with a scale a row (or a row and `group` columns), as the engine rounds activations
            shape = x.shape
            y = x.float().reshape(shape[0], -1, group) if group else x.float().reshape(shape[0], 1, -1)
            top = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
            return ((y * (127.0 / top)).round().clamp(-127, 127) * (top / 127.0)).reshape(shape).to(x.dtype)

        def split(x, count=128):
            # the same, with the `count` channels that peak highest rounded on a scale of their own
            y = x.float()
            peaks = y.abs().amax(dim=0)
            chosen = torch.zeros_like(peaks, dtype=torch.bool)
            chosen[torch.topk(peaks, count).indices] = True
            out = torch.empty_like(y)
            out[:, chosen] = rows(y[:, chosen])
            out[:, ~chosen] = rows(y[:, ~chosen])
            crests.append(float((y.abs().amax(dim=1) / y.pow(2).mean(dim=1).sqrt()).mean()))
            crests.append(float((y[:, ~chosen].abs().amax(dim=1) / y[:, ~chosen].pow(2).mean(dim=1).sqrt()).mean()))
            return out.to(x.dtype)

        crests = []

        def wrap_input(module, quant):
            original = module.forward
            module.forward = lambda x, *a, **k: original(quant(x), *a, **k)
            return lambda: setattr(module, "forward", original)

        # all 50 blocks with the activations' rounding emulated, a scale a row against 128 channels apart
        def through(quant):
            undo = []
            if quant is not None:
                for block in self.model.blocks:
                    for module in (block.attn.qkv_proj, block.attn.out_proj, block.mlp.fc1):
                        undo.append(wrap_input(module, quant))
            x = args["img"].clone()
            for block in self.model.blocks:
                x = block(x, args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=options)
            for u in undo:
                u()
            return x

        want = through(None)
        def clipped(x, sigmas, count=0):
            # a scale a row that reaches `sigmas` times the row's rms (never past its largest); larger values saturate
            y = x.float()
            chosen = torch.zeros(y.shape[1], dtype=torch.bool, device=y.device)
            if count:
                chosen[torch.topk(y.abs().amax(dim=0), count).indices] = True
            out = torch.empty_like(y)
            for part in ((~chosen,) if not count else (chosen, ~chosen)):
                z = y[:, part]
                top = torch.minimum(z.abs().amax(dim=1, keepdim=True), sigmas * z.pow(2).mean(dim=1, keepdim=True).sqrt()).clamp_min(1e-12)
                out[:, part] = (z * (127.0 / top)).round().clamp(-127, 127) * (top / 127.0)
            return out.to(x.dtype)

        for label, quant in (("a scale a row", rows), ("clipped at 8 rms", lambda x: clipped(x, 8.0)), ("clipped at 6 rms", lambda x: clipped(x, 6.0)),
                             ("clipped at 5 rms", lambda x: clipped(x, 5.0)), ("clipped at 4 rms", lambda x: clipped(x, 4.0)),
                             ("128 apart, clipped at 6 rms", lambda x: clipped(x, 6.0, 128)), ("128 apart, clipped at 5 rms", lambda x: clipped(x, 5.0, 128)),
                             ("128 apart, clipped at 4 rms", lambda x: clipped(x, 4.0, 128))):
            got = through(quant)
            logging.info(f"TensorFold H3 sources: 50 blocks, activations rounded with {label}: cosine {cosine(got, want):.5f}, "
                         f"relative error {float((got.float() - want.float()).norm() / want.float().norm()):.4f}")
            del got
        got = args["img"].clone()
        self.run({**args, "img": got}, dense=True)
        logging.info(f"TensorFold H3 sources: 50 blocks, the engine: cosine {cosine(got, want):.5f}, "
                     f"relative error {float((got.float() - want.float()).norm() / want.float().norm()):.4f}")
        del want, got
        crests.clear()
        for index in ():
            block = self.model.blocks[index]
            # the stream as this block meets it
            x = args["img"].clone()
            for before in self.model.blocks[:index]:
                x = before(x, args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=options)
            want = block(x.clone(), args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=options)
            added = float((want.float() - x.float()).norm())
            xf = x.float()
            crest = float((xf.abs().amax(dim=1) / xf.pow(2).mean(dim=1).sqrt()).mean())
            peaks = xf.abs().amax(dim=0)
            top = torch.topk(peaks, 64).values
            report = [f"block {index}: stream crest {crest:.1f} (a row's largest over its rms), the 64 largest channels peak at "
                      f"{float(top.min()):.1f} to {float(top.max()):.1f} against a median channel peak of {float(peaks.median()):.2f}"]
            tests = {"q k v input rows": [(block.attn.qkv_proj, rows)], "attention out input rows": [(block.attn.out_proj, rows)],
                     "mlp input rows": [(block.mlp.fc1, rows)], "all three": [(block.attn.qkv_proj, rows), (block.attn.out_proj, rows), (block.mlp.fc1, rows)],
                     "all three, 128 channels apart": [(block.attn.qkv_proj, split), (block.attn.out_proj, split), (block.mlp.fc1, split)],
                     "all three, 32 apart": [(block.attn.qkv_proj, lambda x: split(x, 32)), (block.attn.out_proj, lambda x: split(x, 32)), (block.mlp.fc1, lambda x: split(x, 32))]}
            for label, wraps in tests.items():
                undo = [wrap_input(module, quant) for module, quant in wraps]
                got = block(x.clone(), args["t_emb"], args["mod_segments"], args["rope_freqs"], transformer_options=options)
                for u in undo:
                    u()
                report.append(f"{label} {float((got.float() - want.float()).norm()) / added:.4f}")
            got = x.clone()
            self.run({**args, "img": got}, blocks=1, dense=True) if index == 0 else None
            if index == 0:
                report.append(f"the engine's whole block {float((got.float() - want.float()).norm()) / added:.4f}")
            report.append("crest of q k v / attention out / mlp inputs, whole then without the 128: " + " ".join(f"{c:.1f}" for c in crests[:6]))
            crests.clear()
            logging.info("TensorFold H3 sources: " + "; ".join(report))
            del want, x

    def first(self, args, extra):
        if os.environ.get("TF_H3_CHECK") == "2" and not self.checked:
            self.checked = True
            self.sources(args)
        if os.environ.get("TF_H3_CHECK") == "1" and not self.checked:
            self.checked = True
            self.check(args)
        profile = os.environ.get("TF_H3_PROFILE") == "1"
        handle = engine(self.path)
        if profile:
            library().tf_h3_profile(handle, 1)
        mark = time.perf_counter()
        took, keep, tiles = self.run(args)
        self.passes += 1
        if self.passes == 1:
            mode = f"{keep} video tiles kept a query tile" if keep else "dense attention"
            logging.info(f"TensorFold H3: {args['img'].shape[0]} rows in {tiles} tiles, {mode}, first pass {took:.2f} s")
        if profile:
            spent, names = (ctypes.c_double * 16)(), (ctypes.c_char_p * 16)()
            n = library().tf_h3_spent(handle, ctypes.byref(spent), ctypes.byref(names))
            parts = ", ".join(f"{names[i].decode()} {spent[i]:.2f}" for i in range(n))
            logging.info(f"TensorFold H3 pass {self.passes}: {time.perf_counter() - mark:.2f} s ({parts})")
            library().tf_h3_profile(handle, 0)
        return {"img": args["img"]}


def passthrough(args, extra):
    return {"img": args["img"]}


# Environment: TF_H3_LIB (the library), TF_H3_CHECK=1|2 (compare with ComfyUI's blocks), TF_H3_PROFILE=1 (timing).
class TensorFoldH3Blocks:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "unet_name": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "The FastH3 bf16 checkpoint the model was loaded from."}),
            "sparsity": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 0.99, "step": 0.05,
                                   "tooltip": "FastH3's sparse attention: the share of video tiles a tile leaves out. 0 is dense."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "model/patch"

    def patch(self, model, unet_name, sparsity):
        diffusion_model = model.get_model_object("diffusion_model")
        if not isinstance(diffusion_model, MiniMaxH3Model):
            raise ValueError("TensorFold H3 needs a MiniMax H3 / FastH3 model")
        path = folder_paths.get_full_path_or_raise("diffusion_models", unet_name)
        m = model.clone()
        state = Blocks(diffusion_model, path, sparsity, take_block_patches(m))
        if state.patches:
            # what ComfyUI patches differs now from the model it came from: its loaded copy must not count as this one
            m.patches_uuid = uuid.uuid4()
            logging.info(f"TensorFold H3: {len(state.patches)} projections carry a LoRA; the engine merges it")
        for i in range(len(diffusion_model.blocks)):
            m.set_model_patch_replace(state.first if i == 0 else passthrough, "dit", "double_block", i)

        def no_prefetch(executor, x, timestep, context, transformer_options={}, **kwargs):
            # ComfyUI would stream its own copy of every block's weights onto the GPU ahead of the block
            transformer_options["prefetch_dynamic_vbars"] = False
            return executor(x, timestep, context, transformer_options, **kwargs)

        m.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, "tensorfold_h3", no_prefetch)

        def reset(model_patcher):
            state.checked = False
            state.passes = 0

        m.add_callback_with_key(comfy.patcher_extension.CallbacksMP.ON_CLEANUP, "tensorfold_h3", reset)
        return (m,)


NODE_CLASS_MAPPINGS = {"TensorFoldH3Blocks": TensorFoldH3Blocks}
NODE_DISPLAY_NAME_MAPPINGS = {"TensorFoldH3Blocks": "TensorFold H3 Blocks (CUDA int8)"}
