"""Flash Next tensor parallel on one GPU: two ranks in two threads, their all-gathers through host memory.

A small random checkpoint in the MLX layout (the real head sizes, a DeltaNet layer, an attention layer, 64
experts, an MTP head) is written to disk and loaded three times: whole, and as rank 0 and rank 1 of two
(``weights.load(tp=...)``, the slicing two machines use). Checks: both ranks emit the same tokens;
MTP-drafted decoding emits serial decoding's tokens (greedy and sampled, full and draft vocabularies); window
rows give serial steps' bits; the two ranks' logits agree with the one-GPU model's to rounding.
"""

import json
import struct
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.weights import load  # noqa: E402

D, S, LOW, E, W, V = 1024, 4, 320, 64, 128, 4096
HEADS, KV, HD, NK, NV, DK, DV, IH, ID = 24, 2, 256, 16, 48, 128, 128, 4, 128
PROMPT = [5, 17, 99, 250, 1023, 7, 64, 300, 11, 12, 13]


def _save(path, tensors: dict) -> None:
    kinds = {torch.int32: "I32", torch.bfloat16: "BF16", torch.float32: "F32"}
    header, blobs, at = {}, [], 0
    for name, t in tensors.items():
        raw = t.contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        header[name] = {"dtype": kinds[t.dtype], "shape": list(t.shape), "data_offsets": [at, at + len(raw)]}
        blobs.append(raw)
        at += len(raw)
    head = json.dumps(header).encode()
    head += b" " * (-len(head) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(head)))
        f.write(head)
        for blob in blobs:
            f.write(blob)


def _checkpoint(root) -> None:
    g = torch.Generator().manual_seed(7)
    t: dict = {}

    def q(name, n, k, lead=(), scale=0.02):
        words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=g, dtype=torch.int64)
        s = (torch.rand((*lead, n, k // 32), generator=g) * scale / 8 + scale / 64).to(torch.bfloat16)
        t[name + ".weight"] = words.to(torch.int32)
        t[name + ".scales"] = s
        t[name + ".biases"] = (-(s.float() * 7.5)).to(torch.bfloat16)          # centred values

    def norm(name, n):
        t[name] = (1 + 0.05 * torch.randn((n,), generator=g)).to(torch.bfloat16)

    def hc(base, inject):
        q(base + ".input_mix_weight_down", LOW, S * D)
        if inject:
            q(base + ".block_inject_weight", S, S * D)
        q(base + ".input_mix_weight_up", S * D, LOW)
        norm(base + ".hc_norm.weight", S * D)

    def attention(base):
        q(base + ".q_proj", HEADS * 2 * HD, D)
        q(base + ".k_proj", KV * HD, D)
        q(base + ".v_proj", KV * HD, D)
        q(base + ".indexer.index_qk_proj", (IH + 1) * ID, D)
        q(base + ".o_proj", D, HEADS * HD)
        norm(base + ".q_norm.weight", HD)
        norm(base + ".k_norm.weight", HD)
        norm(base + ".indexer.q_layernorm.weight", ID)
        norm(base + ".indexer.k_layernorm.weight", ID)

    def gdn(base):
        conv_dim = 2 * NK * DK + NV * DV
        q(base + ".in_proj_qkv", conv_dim, D)
        q(base + ".in_proj_z", NV * DV, D)
        q(base + ".in_proj_b", NV, D)
        q(base + ".in_proj_a", NV, D)
        q(base + ".out_proj", D, NV * DV)
        t[base + ".conv1d.weight"] = (torch.randn((conv_dim, 4, 1), generator=g) * 0.3).to(torch.bfloat16)
        t[base + ".A_log"] = torch.randn((NV,), generator=g) * 0.5
        t[base + ".dt_bias"] = torch.randn((NV,), generator=g) * 0.5
        norm(base + ".norm.weight", DV)

    def moe(base):
        t[base + ".gate.weight"] = (torch.randn((E, D), generator=g) * 0.05).to(torch.bfloat16)
        q(base + ".shared_expert_gate", 1, D, scale=0.2)
        q(base + ".switch_mlp.gate_proj", W, D, (E,))
        q(base + ".switch_mlp.up_proj", W, D, (E,))
        q(base + ".switch_mlp.down_proj", D, W, (E,))
        q(base + ".shared_expert.gate_proj", W, D)
        q(base + ".shared_expert.up_proj", W, D)
        q(base + ".shared_expert.down_proj", D, W)

    def layer(base, linear):
        hc(base + ".attn_hyper_connection", True)
        hc(base + ".mlp_hyper_connection", True)
        gdn(base + ".linear_attn") if linear else attention(base + ".self_attn")
        moe(base + ".mlp")

    q("model.embed_tokens", V, D, scale=0.5)
    layer("model.layers.0", True)
    layer("model.layers.1", False)
    hc("model.hyper_connection_mixer", False)
    q("lm_head", V, D, scale=0.2)
    norm("mtp.pre_fc_norm_embedding.weight", D)
    norm("mtp.pre_fc_norm_hidden.weight", S * D)
    q("mtp.fc_embedding", D, D)
    q("mtp.fc_hidden", D, D)
    layer("mtp.layers.0", False)
    hc("mtp.hyper_connection_mixer", False)
    _save(root / "model.safetensors", t)
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": {name: "model.safetensors" for name in t}}))
    config = {
        "model_type": "qwen4_exp", "hidden_size": D, "num_hidden_layers": 2,
        "layer_types": ["linear_attention", "full_attention"], "vocab_size": V, "rms_norm_eps": 1e-6,
        "num_attention_heads": HEADS, "num_key_value_heads": KV, "head_dim": HD,
        "rope_parameters": {"rope_theta": 1e7, "partial_rotary_factor": 0.25},
        "linear_num_key_heads": NK, "linear_num_value_heads": NV, "linear_key_head_dim": DK,
        "linear_value_head_dim": DV, "linear_conv_kernel_dim": 4, "num_experts": E, "num_experts_per_tok": 10,
        "moe_intermediate_size": W, "shared_expert_intermediate_size": W, "hc_count": S, "hc_lowrank": LOW,
        "indexer_n_heads": IH, "indexer_head_dim": ID, "indexer_budget": 2048, "indexer_compress_ratio": 4,
        "eos_token_id": 0, "quantization": {"group_size": 32, "bits": 4},
    }
    (root / "config.json").write_text(json.dumps(config))
    (root / "draft_ids.txt").write_text("\n".join(str(i) for i in range(0, V, 2)) + "\n")


class _Hub:
    def __init__(self, world: int) -> None:
        self.world = world
        self.slots: list = [None] * world
        self.barrier = threading.Barrier(world, timeout=300)


class _ThreadComm:
    """``comm.NCCL``'s all_gather for ranks that are threads of one process on one GPU."""

    def __init__(self, hub: _Hub, rank: int) -> None:
        self.hub, self.rank, self.world = hub, rank, hub.world

    def all_gather(self, send: torch.Tensor, recv: torch.Tensor) -> None:
        if recv.numel() != send.numel() * self.world or send.dtype != recv.dtype:
            raise ValueError("all_gather: recv must hold world x send of the same dtype")
        torch.cuda.current_stream().synchronize()
        self.hub.slots[self.rank] = send
        self.hub.barrier.wait()
        n = send.numel()
        flat = recv.view(-1)
        for r in range(self.world):
            flat[r * n:(r + 1) * n].copy_(self.hub.slots[r].reshape(-1))
        torch.cuda.current_stream().synchronize()
        self.hub.barrier.wait()

    def barrier(self) -> None:
        self.hub.barrier.wait()


def _run_ranks(fn, engines: list) -> list:
    """fn(rank, engine) on every rank at once (threads); results in rank order."""

    hub = _Hub(len(engines))
    for r, e in enumerate(engines):
        e.w.comm = _ThreadComm(hub, r)
    results: list = [None] * len(engines)
    errors: list = []

    def body(r: int) -> None:
        try:
            with torch.no_grad():
                results[r] = fn(r, engines[r])
        except BaseException as exc:        # noqa: BLE001  (reported below; unblock the other rank)
            errors.append(exc)
            hub.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(len(engines))]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    if errors:
        raise errors[0]
    return results


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    root = tmp_path_factory.mktemp("flashnext_tp")
    _checkpoint(root)
    return root


@pytest.fixture(scope="module")
def models(checkpoint):
    root = checkpoint
    single = load(root)
    ranks = [load(root, tp=(r, 2)) for r in range(2)]
    drafts = [load(root, tp=(r, 2), draft_vocab=str(root / "draft_ids.txt")) for r in range(2)]
    return single, ranks, drafts


def test_ranks_split_heads_experts_and_vocabulary(models):
    single, ranks, _ = models
    for r, w in enumerate(ranks):
        assert w.meta["world"] == 2 and w.meta["rank"] == r
        assert (w.cfg.heads, w.cfg.kv_heads, w.cfg.nk, w.cfg.nv) == (HEADS // 2, KV // 2, NK // 2, NV // 2)
        assert w.cfg.moe_width == W // 2 and w.head.n == V // 2
    assert single.head.n == V


def test_tp_logits_agree_with_one_gpu_to_rounding(models):
    single, ranks, _ = models
    e1 = Engine(single, capacity=1024, max_rows=8, prefill_rows=16)
    ref = forward(single, e1.st, e1.buf, PROMPT)[:len(PROMPT)].float().clone()
    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16) for w in ranks]
    parts = _run_ranks(lambda r, e: forward(e.w, e.st, e.buf, PROMPT)[:len(PROMPT)].float().clone(), engines)
    got = torch.cat(parts, dim=1)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=1)
    assert float(cos.min()) > 0.999, cos
    assert float((got - ref).abs().max()) < 0.05 * float(ref.abs().max())


def test_tp_windows_match_serial_steps_and_prefix_commits_continue(models):
    _, ranks, _ = models
    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16) for w in ranks]
    nxt = [401, 33, 2048, 5, 77, 1500, 9, 10, 11]

    def body(r, e):
        w = e.w
        forward(w, e.st, e.buf, PROMPT)
        commit(w, e.st, e.buf, len(PROMPT), len(PROMPT))
        serial = e.st.clone()
        steps = []
        for t in nxt:
            steps.append(forward(w, serial, e.buf, [t])[0].clone())
            commit(w, serial, e.buf, 1, 1)
        bad = []
        for R in (2, 3, 4, 8):
            for keep in sorted({1, max(1, R // 2), R}):
                st = e.st.clone()
                lg = forward(w, st, e.buf, nxt[:R])
                bad += [(R, row) for row in range(R) if not torch.equal(lg[row], steps[row])]
                commit(w, st, e.buf, R, keep)
                if not torch.equal(forward(w, st, e.buf, [nxt[keep]])[0], steps[keep]):
                    bad.append((R, "keep", keep))
        return bad

    for r, bad in enumerate(_run_ranks(body, engines)):
        assert not bad, (r, bad)


@pytest.mark.parametrize("sampling", [None, Sampling(seed=1234, top_k=20, top_p=0.95)])
def test_tp_mtp_drafts_give_serial_tokens_on_both_ranks(models, sampling):
    _, ranks, drafts = models

    def body(r, e):
        first = prefill(e, PROMPT, sampling)
        out = {"serial": serial_decode(e, first, 24, sampling).tokens}
        for name, depth, conf in (("d1", 1, 0.0), ("d3", 3, 0.0), ("d5", 5, 0.0), ("d5c", 5, 0.001),
                                  ("d6c30", 6, 0.3), ("d7c90", 7, 0.9)):
            prefill(e, PROMPT, sampling)
            got = mtp_decode(e, first, 24, sampling, depth=depth, confidence=conf)
            out[name] = got.tokens
            out[name + "_accepted"] = got.accepted
            out[name + "_drafted"] = got.drafted
        return out

    for group in (ranks, drafts):
        engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16) for w in group]
        a, b = _run_ranks(body, engines)
        assert a == b
        for name in ("d1", "d3", "d5", "d5c", "d6c30", "d7c90"):
            assert a[name] == a["serial"], (name, group is drafts)
        assert a["d5c_drafted"] <= a["d5_drafted"] and a["d7c90_drafted"] == 0


@pytest.mark.parametrize("sampling", [None, Sampling(seed=77, top_k=20, top_p=0.95)])
def test_tp_kept_drafts_give_serial_tokens(models, sampling, monkeypatch):
    """The random MTP head's drafts all miss, so rounds here draft serial decoding's own tokens with every fifth
    position wrong: windows keep runs of 1-5 drafts (the verify, commit and replay paths across ranks)."""

    from tensorfold.families.qwen4_exp.cuda import decode

    _, ranks, _ = models
    real = decode.draft
    L = len(PROMPT)

    def oracle(e, streams, next_tokens, position, count, sampling_, confidence=0.0):
        real(e, streams, next_tokens, position, count, sampling_, confidence)
        ref = e.reference
        return [ref[p - L] if p % 5 != 3 else (ref[p - L] + 1) % V for p in range(position, position + count)]

    monkeypatch.setattr(decode, "draft", oracle)

    def body(r, e):
        first = prefill(e, PROMPT, sampling)
        e.reference = serial_decode(e, first, 40, sampling).tokens
        out = {"serial": e.reference}
        for name, depth in (("d3", 3), ("d5", 5), ("d6", 6)):
            prefill(e, PROMPT, sampling)
            got = mtp_decode(e, first, 40, sampling, depth=depth, confidence=0.0)
            out[name] = got.tokens
            out[name + "_keeps"] = got.keeps
        return out

    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16) for w in ranks]
    a, b = _run_ranks(body, engines)
    assert a == b
    for name in ("d3", "d5", "d6"):
        assert a[name] == a["serial"], name
        assert max(a[name + "_keeps"]) >= 3, (name, a[name + "_keeps"])


def test_tp_candidates_gathered_in_the_step_match_the_eager_gather(models):
    """The candidates a forward gathers inside its graph pick the same tokens (and probabilities) as gathering
    after the step, greedy and sampled, for the main head and the draft head."""

    from tensorfold.families.qwen4_exp.cuda.decode import choose_gathered, tp_sample_rows

    _, _, drafts = models
    engines = [Engine(w, capacity=1024, max_rows=8, prefill_rows=16) for w in drafts]
    samplings = (None, Sampling(seed=4321, top_k=20, top_p=0.95))

    def body(r, e):
        w = e.w
        lg = forward(w, e.st, e.buf, PROMPT)[:len(PROMPT)]
        rows = len(PROMPT)
        positions = list(range(10, 10 + rows))
        out = []
        for s in samplings:
            a = choose_gathered(w, e.buf.cand_all, rows, positions, s, with_prob=True)
            b = tp_sample_rows(w, lg, positions, s, offset=w.meta["vocab_offset"], with_prob=True)
            out.append((a[0] == b[0], max(abs(x - y) for x, y in zip(a[1], b[1]))))
        return out

    for r, res in enumerate(_run_ranks(body, engines)):
        for same, dp in res:
            assert same and dp < 1e-5, (r, same, dp)



def _fake_nccl(monkeypatch, hub):
    """comm.NCCL for two ranks that are threads of this process: a real TCP store on localhost for the requests,
    all-gathers through host memory."""

    import socket
    from datetime import timedelta

    from torch.distributed import TCPStore

    from tensorfold.families.qwen4_exp.cuda import comm as comm_mod

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

    class FakeNCCL:
        def __init__(self, rank, world, master, port_):
            self.rank, self.world = rank, world
            self.store = TCPStore("127.0.0.1", port, 2, rank == 0, timeout=timedelta(seconds=120))
            self.inner = _ThreadComm(hub, rank)

        def all_gather(self, send, recv):
            self.inner.all_gather(send, recv)

        def barrier(self):
            hub.barrier.wait()

    monkeypatch.setattr(comm_mod, "NCCL", FakeNCCL)


def test_tp_server_ranks_share_requests_and_stream_serial_tokens(checkpoint, models, monkeypatch):
    """engine.FlashNextEngine with two ranks: rank 0 hands each request to rank 1 through the TCP store, both
    decode it in lockstep, rank 0 streams the tokens tensor-parallel serial decoding gives, and rank 1 leaves
    when rank 0 shuts down."""

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    _, ranks, _ = models
    sampling = Sampling(seed=1234, top_k=20, top_p=0.95)
    refs = _run_ranks(lambda r, e: serial_decode(e, prefill(e, PROMPT, sampling), 20, sampling).tokens,
                      [Engine(w, capacity=512, max_rows=8, prefill_rows=16) for w in ranks])
    hub = _Hub(2)
    _fake_nccl(monkeypatch, hub)
    engines: list = [None, None]
    errors: list = []

    def build(r):
        try:
            engines[r] = FlashNextEngine(checkpoint, depth=4, confidence=0.001, draft_vocab=None, max_len=512, tp=2,
                                         rank=r, master="127.0.0.1", prefetch=False, graphs=False)
            if r == 1:
                engines[1].follow()
        except BaseException as exc:            # noqa: BLE001
            errors.append(exc)
            hub.barrier.abort()

    follower = threading.Thread(target=build, args=(1,))
    follower.start()
    build(0)
    assert not errors, errors
    def ask(prompt, samp, **kw):
        got: list = []
        stats = engines[0].generate(prompt, 20, samp, lambda new: got.extend(new), **kw)
        return got, stats

    with torch.no_grad():
        got, _ = ask(PROMPT, sampling)
        serial, serial_stats = ask(PROMPT, sampling, draft=False)      # one token a round on both ranks
        prompt2 = PROMPT + got + [7, 8, 9]
        warm, warm_stats = ask(prompt2, sampling)                      # resumes from the reply on both ranks
        cold, _ = ask(prompt2, sampling, draft=False)
        greedy, _ = ask(PROMPT, None)
    engines[0].shutdown()
    follower.join(timeout=120)
    assert not follower.is_alive() and not errors, errors
    ref = refs[0]
    eos = [i for i, t in enumerate(ref) if t in engines[0].eos]
    assert got == (ref[:eos[0] + 1] if eos else ref)
    assert serial == got and serial_stats["drafts"] is False
    assert warm_stats["cached"] >= len(PROMPT) + len(got) - 1 and warm == cold
    assert len(greedy) >= 1


def test_tp_ranks_started_with_different_settings_refuse_to_start(checkpoint, monkeypatch):
    """Two ranks with different draft rules would fall out of step: both raise before loading the weights."""

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    hub = _Hub(2)
    _fake_nccl(monkeypatch, hub)
    raised: list = [None, None]

    def build(r):
        try:
            FlashNextEngine(checkpoint, depth=4 if r == 0 else 3, draft_vocab=None, max_len=512, tp=2, rank=r,
                            master="127.0.0.1", prefetch=False, graphs=False)
        except RuntimeError as exc:
            raised[r] = exc

    threads = [threading.Thread(target=build, args=(r,)) for r in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout=120)
    assert all(isinstance(x, RuntimeError) and "different settings" in str(x) for x in raised), raised
