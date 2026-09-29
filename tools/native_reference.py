"""Numerical oracle for the native Qwen port (no native process launching).

Generate reference logits with this checkout's actual lane forward, or compare
already generated .npy files. Artifacts live under build/native-checks.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def vision_fixture(model_dir, output, height, width, image_fixture=False, image_format="PNG", image_alpha=False, image_orientation=1, image_only=False, image_mode=None):
    from native_runtime import require_mlx
    require_mlx()
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_vlm.models.qwen3_5.config import VisionConfig
    from mlx_vlm.models.qwen3_5.vision import VisionModel
    from tensorfold.vision.qwen_checkpoint import load_vision_weights, quantization_predicate, vision_tensors
    config = json.loads((Path(model_dir) / "config.json").read_text())
    out = Path(output)
    (out / "python").mkdir(parents=True, exist_ok=True)
    if image_fixture:
        from PIL import Image
        from tensorfold.vision.qwen_processing import QwenImageProcessor
        import base64
        from tensorfold.vision.images import ImageSource, load_images
        image = Image.fromarray(np.random.default_rng(314159).integers(0, 256, (height, width, 4 if image_alpha else 3), dtype=np.uint8))
        if image_mode:
            image = image.convert(image_mode)
        exif = Image.Exif()
        exif[274] = image_orientation
        image_path = out / ("image." + image_format.lower())
        image.save(image_path, format=image_format, exif=exif)
        image = load_images([ImageSource("data:image/" + image_format.lower() + ";base64," + base64.b64encode(image_path.read_bytes()).decode())])[0].to_pil()
        processor = QwenImageProcessor.from_directory(model_dir).processor
        processed = processor(images=[image], max_pixels=4096 * 1024, min_pixels=65536)
        pixels = np.asarray(processed["pixel_values"], dtype=np.float32)
        _, height, width = map(int, processed["image_grid_thw"][0])
    else:
        pixels = np.random.default_rng(314159).uniform(-1, 1, (height * width, 1536)).astype(np.float32)
    np.save(out / "pixels.npy", pixels)
    (out / "grid.json").write_text(json.dumps(dict(height=height, width=width)))
    if image_only:
        print(f"Saved upstream image preprocessing in {out}")
        return
    tower = VisionModel(VisionConfig.from_dict(config["vision_config"]))
    weights = tower.sanitize(load_vision_weights(vision_tensors(Path(model_dir)), mx))
    nn.quantize(tower, class_predicate=quantization_predicate(config, weights))
    tower.load_weights(list(weights.items()), strict=True)
    tower.eval()
    def save(name, x):
        mx.eval(x)
        np.save(out / "python" / f"{name}.npy", np.array(x.astype(mx.float32)))
    grid = mx.array([[1, height, width]], dtype=mx.int32)
    h = tower.patch_embed(mx.array(pixels).astype(tower.patch_embed.proj.weight.dtype))
    save("patch", h)
    position = tower.fast_pos_embed_interpolate(grid)
    save("position", position)
    h = h + position
    freq = tower.rot_pos_emb(grid)
    save("frequencies", mx.concatenate([freq, freq], -1).reshape(1, height * width, 1, 72))
    cu = mx.array([0, height * width], dtype=mx.int32)
    for i, block in enumerate(tower.blocks):
        h = block(h, cu, freq)
        save(f"block-{i}", h)
    save("embeddings", tower.merger(h))
    print(f"Saved upstream vision stages in {out}")


def image_http_fixtures(output):
    import ipaddress
    import random
    from tensorfold.vision.images_http import _public_ip, _url, ImageInputError
    addresses = {"invalid", "127.1", "01.2.3.4", "::ffff:8.8.8.8", "2002:0808:0808::1", "168.63.129.16"}
    for cls in (ipaddress.IPv4Address, ipaddress.IPv6Address):
        constants = cls._constants
        blocks = [*constants._private_networks, *constants._private_networks_exceptions]
        if hasattr(constants, "_reserved_networks"):
            blocks += constants._reserved_networks
        else:
            blocks += [constants._public_network, constants._multicast_network]
        for block in blocks:
            for boundary in (int(block.network_address), int(block.broadcast_address)):
                for offset in (-1, 0, 1):
                    if 0 <= boundary + offset < 1 << cls.max_prefixlen:
                        addresses.add(str(cls(boundary + offset)))
        rng = random.Random(271828)
        for _ in range(1024):
            addresses.add(str(cls(rng.getrandbits(cls.max_prefixlen))))
    urls = [
        "https://example.com", "https://example.com:443/a?x=1", "https://example.com/a#",
        "HTTPS://EXAMPLE.COM/a", "https://bücher.example/æ?q=ø", "https://faß.example/a",
        "https://[2606:4700:4700::1111]/a", "https://[2606:ABCD::1]/a",
        "https://example.com/%2f?q=%20", "https://example.com/a/../b", "https://example.com/a?x=[a]",
        "http://example.com/a", "https://user@example.com/a", "https://user:pass@example.com/a",
        "https://example.com:8443/a", "https://example.com/#x", "https://example.com\\@localhost/a",
        "https://example.com/a\n", "https://%31%32%37.0.0.1/a", "https://[fe80::1%25en0]/a",
        "https://localhost./a", "https://METADATA.GOOGLE.INTERNAL/a", "https://instance-data/a",
        "https:///a", "https://example.com/" + "x" * 4096,
        "https://example.com:/a", "https://example.com/a?", "https://example.com/" + "ø" * 3000,
        "https://" + ".".join(["ø" * 30] * 6) + "/a", "https://example.com／bad/a",
        "https://example.com:+443/a", "https://😀.example/a", "https://under_score.example/a",
    ]
    cases = []
    for value in urls:
        try:
            host, _, target = _url(value, 4096)
            canonical = "https://" + ("[" + host + "]" if ":" in host else host) + target
        except ImageInputError:
            canonical = None
        cases.append(dict(value=value, canonical=canonical))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(ips=[dict(value=value, public=_public_ip(value)) for value in sorted(addresses)], urls=cases)))
    print(f"Saved {len(addresses)} upstream image address and {len(cases)} URL fixtures")


def verify_capture(model_dir, draft_dir, report_path):
    import struct
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from tensorfold.families.qwen3_5 import load_lane_model
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm, lane_tree
    from tensorfold.engine.topk import topk_rows
    from tensorfold.drafters.dflash_drafter import DFlashDrafter
    report = json.loads(Path(report_path).read_text())
    base = Path(report['draft_capture'])
    history = report['prompt_tokens'] + report['tokens']
    meta = json.loads(base.with_suffix('.json').read_text())
    for key in ('seed', 'temperature', 'top_k', 'top_p'):
        assert meta[key] == (report[key] if report['temperature'] else None), (key, meta[key])
    assert meta['first_position'] == 0 and meta['width'] == 5120
    targets = {}
    data = base.with_suffix('.logits').read_bytes()
    at = 0
    while at < len(data):
        count, k = struct.unpack_from('<ii', data, at)
        at += 8
        positions = np.frombuffer(data, '<i8', count=count, offset=at)
        at += count * 8
        ids = np.frombuffer(data, '<i4', count=count * k, offset=at).reshape(count, k)
        at += count * k * 4
        vals = np.frombuffer(data, '<f4', count=count * k, offset=at).reshape(count, k)
        at += count * k * 4
        for position, ids_row, vals_row in zip(positions, ids, vals):
            assert int(position) not in targets, 'Duplicate target capture position'
            targets[int(position)] = ids_row, vals_row
    assert at == len(data)
    assert sorted(targets) == list(range(len(report['prompt_tokens']), len(history))), 'Missing or uncommitted target capture positions'
    contexts = []
    data = base.with_suffix('.bin').read_bytes()
    at, end = 0, 0
    while at < len(data):
        start, rows, width = struct.unpack_from('<qii', data, at)
        at += 16
        bits = np.frombuffer(data, '<u2', count=rows * width, offset=at).reshape(rows, width)
        at += rows * width * 2
        tokens = np.frombuffer(data, '<i4', count=rows, offset=at).tolist()
        at += rows * 4
        assert start == end and width == meta['width'] and rows > 0
        assert tokens == history[start:start + rows], 'Rejected tree row entered capture'
        contexts.append((start, tokens, bits))
        end += rows
    assert at == len(data) and len(history) - 1 <= end <= len(history)
    regular = report['prefill_mode'] == 'regular'
    if regular:
        from tensorfold.families.qwen3_5 import load
        family, _ = load(Path(model_dir), lane_kernels='on')
        model = family.inner
    else:
        model, _ = load_lane_model(Path(model_dir))
        lane_qmm.install(model, rows=128, tile=True, wide=True)
    drafter = DFlashDrafter(model, str(draft_dir), bits=4)
    lane_qmm.install(drafter.model, rows=128, tile=True, wide=True)
    cache = make_prompt_cache(model)
    core, head = model.language_model.model, model.language_model.lm_head
    checked, prefill_start = 0, -1
    for start, tokens, bits in contexts:
        prompt_chunk = regular and start < len(report['prompt_tokens'])
        if prompt_chunk:
            chunk_start = start // 2048 * 2048
            if prefill_start != chunk_start:
                inputs = report['prompt_tokens'][chunk_start:chunk_start + 2048]
                hidden = family.prefill(mx.array([inputs], dtype=mx.uint32), cache)
                prefill_logits, prefill_taps = family.head(hidden[:, -1:]), drafter.taps()
                mx.eval(prefill_logits, prefill_taps)
                prefill_start = chunk_start
            logits = prefill_logits
            taps = prefill_taps[:, start - chunk_start:start - chunk_start + len(tokens)]
        else:
            logits, record = lane_tree.tree_forward(core, head, tokens, list(range(-1, len(tokens) - 1)), cache, start)
            taps = drafter.taps()
        projected = drafter.model.hidden_norm(drafter.model.fc(taps))
        mx.eval(logits, projected)
        actual = np.array(projected[0].view(mx.uint16))
        assert np.array_equal(bits, actual), f'Projected capture features differ at {start}: {np.count_nonzero(bits != actual)} BF16 values'
        for row in range(len(tokens)):
            position = start + row + 1
            if position not in targets:
                continue
            ids, vals = topk_rows(logits[:, -1:] if prompt_chunk else logits[:, row:row + 1], len(targets[position][0]))
            mx.eval(ids, vals)
            assert np.array_equal(np.array(ids)[0], targets[position][0]), f'Target candidate IDs differ at {position}'
            assert np.array_equal(np.array(vals)[0], targets[position][1]), f'Target logits differ at {position}'
            checked += 1
        if not prompt_chunk:
            lane_tree.commit_tree(cache, record, list(range(len(tokens))), len(tokens), start)
    assert checked == len(targets)
    print(f'PASS: {end} committed context rows and {checked} target distributions match upstream exactly; capture positions and metadata agree with generation')


def capture_fixtures(output):
    from types import SimpleNamespace
    from tensorfold.drafters import dflash_proposer
    from tensorfold.drafters.dflash_proposer import DFlashProposer
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=True)
    records = {'.bin': bytearray(), '.logits': bytearray()}
    class Recorder:
        def put(self, item):
            path, record = item
            records[path.suffix] += record
    original = dflash_proposer._capture_writer
    dflash_proposer._capture_writer = lambda: Recorder()
    settings = dict(seed=5678, temperature=0.7, top_k=12, top_p=0.8)
    proposer = DFlashProposer.__new__(DFlashProposer)
    proposer.sampling = SimpleNamespace(**settings)
    proposer.capture_dir = str(folder)
    contexts, targets = [], []
    start = 0
    try:
        for case in range(17):
            count, width = (128 if case == 0 else case + 1), 512
            bits = (np.arange(count * width, dtype=np.uint32) + case * 4093).astype(np.uint16).reshape(count, width)
            tokens = [i * 37 % 248320 for i in range(start, start + count)]
            proposer._captured = start, bits
            proposer._write_capture([0] * start + tokens)
            contexts.append(dict(start=start, width=width, tokens=tokens, bits=bits.ravel().tolist()))
            k = case % 16 + 1
            positions = [0x100000007 + start + i for i in range(case + 1)]
            ids = np.arange(len(positions) * k, dtype=np.int32).reshape(len(positions), k)
            logits = (ids.astype(np.float32) - 100) / 8
            if case == 0:
                logits[0, 0] = -0.0
            proposer.capture_target(positions, ids, logits)
            targets.append(dict(positions=positions, k=k, ids=ids.ravel().tolist(), logits=logits.ravel().tolist()))
            start += count
        metadata = json.loads(proposer._capture_file.with_suffix('.json').read_text())
    finally:
        dflash_proposer._capture_writer = original
    for suffix, data in records.items():
        (folder / ('expected' + suffix)).write_bytes(data)
    (folder / 'fixture.json').write_text(json.dumps(dict(settings=settings, contexts=contexts, targets=targets, metadata=metadata)))
    print(f'Saved {len(contexts)} upstream capture records, including every BF16 bit pattern')


def server_live_fixtures(output):
    import io
    import random
    from types import SimpleNamespace
    from unittest.mock import patch
    from tensorfold.server.live import Meter, ChunkRate, LiveLine, status
    rng = random.Random(30751)
    instant = 0.0
    decoded, prefilled = Meter(clock=lambda: instant), ChunkRate(clock=lambda: instant)
    events = []
    samples = [(0, 1, 1, .5), (0, 2, 2469, 2), (2, 0, 0, 0),
               (2.000001, 0, 0, 0), (3, 999999, 0, -1), (6, 0, 0, 0)]
    for _ in range(1000):
        instant += rng.choice([0, .125, .5, 2, 2.000001, 5])
        samples.append((instant + 6, rng.randrange(5000), rng.randrange(65536), rng.choice([0, -1, .125, .5, 2])))
    for instant, tokens, prefill, seconds in samples:
        decoded.add(tokens)
        prefilled.add(prefill, seconds)
        scheduler = SimpleNamespace(active=rng.randrange(9), waiting=rng.randrange(9), filling=rng.choice([None, object()]), decoded=decoded, prefilled=prefilled)
        events.append(dict(time=instant, tokens=tokens, seconds=seconds, prefill=prefill,
                           connections=scheduler.active + (scheduler.filling is not None) + scheduler.waiting,
                           waiting=scheduler.waiting, decode_rate=decoded.rate(), prefill_rate=prefilled.rate(), text=status(scheduler)))
    out, err = io.StringIO(), io.StringIO()
    text = ''
    live = LiveLine(lambda: text, out)
    lines = []
    for action, text, columns in [
            ('draw', events[1]['text'], 100), ('stderr', 'partial', 100),
            ('draw', 'must not split a log line', 100), ('stdout', '', 100),
            ('draw', 'still incomplete', 100), ('stderr', ' done\n', 100),
            ('draw', events[0]['text'], 24), ('draw', 'é中 · ' * 40, 60),
            ('stdout', 'stdout log\n', 100), ('draw', events[2]['text'], 10),
            ('draw', events[2]['text'], 200), ('stop', '', 100), ('stop', '', 100)]:
        if action == 'draw':
            with patch('shutil.get_terminal_size', return_value=SimpleNamespace(columns=columns)):
                live.draw()
        elif action == 'stop':
            live.stop()
        else:
            live.write(text, out if action == 'stdout' else err)
        lines.append(dict(action=action, text=text, columns=columns, stdout=out.getvalue(), stderr=err.getvalue()))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(events=events, lines=lines)))
    print(f'Saved {len(events)} upstream live rate/status cases and {len(lines)} terminal cases')


def prefill_plan_fixtures(output):
    import random
    from tensorfold.engine.prefill_plan import PrefillPlan
    rng = random.Random(61749)
    cases = []
    for _ in range(1600):
        step = rng.choice((0, 1, 16, 256, 2048))
        minimum = rng.choice((0, 1, 2, 16, 256))
        openers = rng.choice(([], [91], [92, 91, 92]))
        assistant = rng.choice(([], [91], [91, 92], [92, 93, 94]))
        tokens = [rng.randrange(89, 96) for _ in range(rng.choice((0, 1, 15, 16, 17, 255, 256, 257, 2049, 4099)))]
        case = dict(plan=dict(step=step, min_chunk=minimum, openers=openers, assistant=assistant), tokens=tokens,
                    name=None, points=[], starts=[], positions=[], spans=[])
        try:
            plan = PrefillPlan(step, openers, minimum, assistant)
        except ValueError:
            cases.append(case)
            continue
        chunks = plan.chunks(tokens)
        case.update(name=plan.name, points=plan.points(np.asarray(tokens)), starts=chunks.starts)
        positions = {0, 1, len(tokens), len(tokens) + 1, *chunks.starts}
        positions.update(max(0, p - 1) for p in chunks.starts)
        positions.update(p + 1 for p in chunks.starts)
        case['positions'] = [dict(position=p, contains=p in chunks, floor=chunks.floor(p)) for p in sorted(positions)]
        for _ in range(4):
            begin = rng.choice(chunks.starts)
            end = rng.choice([p for p in [*chunks.starts, len(tokens)] if p >= begin])
            case['spans'].append(dict(begin=begin, end=end, chunks=chunks.between(begin, end)))
        cases.append(case)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases))
    print(f'Saved {len(cases)} upstream adaptive prefill plans')


def prompt_cache_fixtures(output):
    import random
    from tensorfold.server.checkpoints import CheckpointStore, choose_checkpoints

    rng = random.Random(418137)
    result = dict(stores=[], checkpoints=[])
    prompts = [list(range(n)) for n in range(25)]
    prompts += [list(range(n)) + [91, 92, 93] for n in range(20)]
    for slots in (1, 3, 8):
        for budget in (None, 1, 120, 1024):
            for pinned_slots in (0, 1, 3):
                store = CheckpointStore(slots, dict, budget_bytes=budget,
                                        sizer=lambda payload: payload['size'], pinned_slots=pinned_slots)
                case = dict(slots=slots, budget=budget, pinned_slots=pinned_slots, operations=[])
                for ident in range(250):
                    prompt = rng.choice(prompts)
                    boundary = dict(step=rng.choice((1, 2, 4, 8)),
                                    starts=rng.choice((None, [1, 3, 8, 16])))
                    usable = lambda n: n in boundary['starts'] if boundary['starts'] is not None else n % boundary['step'] == 0
                    kind = rng.choice(('insert', 'insert', 'insert', 'match', 'match', 'longest', 'evict'))
                    op = dict(kind=kind, prompt=prompt, boundary=boundary)
                    if kind == 'insert':
                        payload = dict(id=ident, size=rng.choice((0, 1, 16, 60, 120, 2048)))
                        previous = prompt + [rng.randrange(100, 110)]
                        pinned = rng.choice((False, False, True))
                        store.admit_oversize = rng.choice((False, True))
                        op.update(payload=payload, previous=previous, pinned=pinned, oversize=store.admit_oversize)
                        store.insert(prompt, payload, last_prompt=previous, pinned=pinned)
                    elif kind == 'match':
                        take = rng.choice((False, False, True))
                        hit = store.match(prompt, usable, take=take)
                        op.update(take=take, hit=None if hit is None else dict(count=hit[0], payload=hit[1], previous=hit[2]))
                    elif kind == 'longest':
                        op['longest'] = store.longest(prompt, usable)
                    else:
                        keep = rng.choice([None, *store._entries])
                        op.update(keep=None if keep is None else keep.tokens, evicted=store.evict_one(keep))
                    op.update(entries=[dict(tokens=e.tokens, payload=e.cache, previous=e.last_prompt, pinned=e.pinned) for e in store._entries],
                              nbytes=store.nbytes, hits=store.hits, misses=store.misses, evictions=store.evictions)
                    case['operations'].append(op)
                result['stores'].append(case)
    for _ in range(1000):
        prompt = rng.choice(prompts)
        previous = rng.choice([None, *prompts])
        history, cached = rng.randrange(30), rng.randrange(30)
        result['checkpoints'].append(dict(prompt=prompt, previous=previous, history=history, cached=cached,
                                          expected=choose_checkpoints(history, cached, previous, prompt)))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result))
    print('Saved 9000 upstream prompt-cache operations and 1000 checkpoint selections')


def memory_fixtures(output):
    from dataclasses import asdict
    import random
    from types import SimpleNamespace
    from tensorfold.engine.memory import StreamMemory, Admission
    from tensorfold.server.memory_budget import CacheMemory, memory_limit_bytes, needed_bytes, largest_context, GIB

    rng = random.Random(81071)
    result = dict(limits=[], caches=[], streams=[], budgets=[])
    for ram in (8 * GIB, 48 * GIB, 128 * GIB, 256 * GIB):
        for recommended in (0, ram // 2, ram, ram * 2):
            for fraction in (0.7, 0.85):
                for override in (None, "1e308", "1000", "110", "2.5", "1e-30", " 64 ", "", "0", "-1", "nan", "inf", "12GB"):
                    mx = SimpleNamespace(device_info=lambda: dict(max_recommended_working_set_size=recommended))
                    try:
                        value = memory_limit_bytes(mx, fraction=fraction, physical_bytes=ram,
                                                   environ={} if override is None else {"TENSORFOLD_MEMORY_LIMIT_GB": override})
                    except ValueError:
                        value = None
                    result["limits"].append(dict(ram=ram, recommended=recommended, fraction=fraction, override=override, result=value))
    for _ in range(600):
        memory = CacheMemory(rng.randrange(0, 1000000), rng.randrange(0, 100000), rng.choice((1, 16, 256, 2048)), rng.choice((0, 128, 16384)))
        tokens = rng.choice((0, 1, 255, 256, 257, 2047, 2048, 2049, rng.randrange(262144)))
        in_flight = rng.randrange(1, 5)
        request = dict(resident_bytes=rng.randrange(20 * GIB), working_bytes=rng.randrange(GIB), cache_copies=rng.randrange(1, 4), reserve_tokens=rng.randrange(8192))
        need = needed_bytes(memory, tokens, **request)
        budget = max(0, need + rng.choice((-1, 0, 1, GIB)))
        window = rng.choice((0, 4096, 262144))
        result["caches"].append(dict(memory=asdict(memory), tokens=tokens, in_flight=in_flight, request=request, budget=budget, window=window,
                                     cache=memory.cache_bytes(tokens), growth=memory.growth_bytes(tokens, in_flight), needed=need,
                                     largest=largest_context(memory, window, budget_bytes=budget, **request)))
    for _ in range(1000):
        first = rng.randrange(64, 2049)
        short = rng.randrange(0, 200000000)
        memory = StreamMemory(first, short, first + rng.randrange(1, 4097), short + rng.randrange(20000000),
                              rng.choice((0.0, 0.5, 131072.0, rng.random() * 65536)), rng.random() * 65536, rng.random() * 8,
                              rng.randrange(100000000), rng.choice((16, 256, 2048)))
        tokens = rng.choice((0, first, first - 1, first + 1, memory.long_tokens, 262144))
        prompt = rng.randrange(0, 8192)
        live = [(rng.randrange(10000), rng.randrange(20000)) for _ in range(rng.randrange(9))]
        used = rng.randrange(20 * GIB)
        admission = Admission(0, memory, used=lambda: used, lanes=rng.choice((1, 2, 4, 8)))
        projected = admission.projected(prompt, tokens, live)
        admission.budget = max(0, projected + rng.choice((-1, 0, 1)))
        result["streams"].append(dict(memory=asdict(memory), lanes=admission.lanes, tokens=tokens, prompt=prompt, used=used, budget=admission.budget,
                                      live=[dict(now=now, most=most) for now, most in live], stream=memory.stream_bytes(tokens),
                                      prefill=memory.prefill_bytes(prompt), projected=projected, admits=admission.admits(prompt, tokens, live),
                                      fitting=admission.fitting(tokens)))
    for fraction in (0.7, 0.85):
        for process in (64 * GIB, 110 * GIB):
            for elsewhere in (0, 8 * GIB, 20 * GIB, 128 * GIB):
                share = process - 3 * GIB
                value = max(0, min(max(int(fraction * 128 * GIB), process) - elsewhere, share))
                result["budgets"].append(dict(ram=128 * GIB, fraction=fraction, process=process, share=share, elsewhere=elsewhere, result=value))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result))


def allocation_fixtures(output):
    import random
    from tensorfold.engine.allocate import allocate, chain_probabilities
    rng = random.Random(271828)
    cases, chains = [], []
    for streams in (0, 1, 2, 3, 8, 32):
        for repeat in range(80):
            fixed = [rng.randrange(1, 9) for _ in range(streams)]
            probabilities = [[rng.choice((0.0, 0.25, 0.5, 0.94, 1.0, rng.random())) for _ in range(rng.randrange(17))] for _ in fixed]
            if repeat % 2:
                probabilities = [sorted(p, reverse=True) for p in probabilities]
            costs = {} if repeat % 4 == 0 else {r: rng.uniform(0.1, 20) for r in range(sum(fixed) + sum(map(len, probabilities)) + 1) if rng.random() < 0.7}
            overhead = rng.choice((0.0, 8.0, 100.0))
            max_rows = rng.randrange(sum(fixed) + sum(map(len, probabilities)) + 1)
            cases.append(dict(fixed=fixed, probabilities=probabilities, costs=[dict(rows=r, ms=ms) for r, ms in costs.items()], overhead=overhead, max_rows=max_rows, expected=allocate(fixed, probabilities, costs, overhead, max_rows)))
    for probabilities in ([[], []], [[0.0, 0.0], [0.0]], [[0.5] * 8, [0.5] * 8], [[0.0, 1.0], [0.0, 1.0]], [[1.0] * 15] * 32):
        fixed = [1] * len(probabilities)
        for max_rows in range(sum(fixed) + sum(map(len, probabilities)) + 2):
            for costs in ({}, {r: float(r) for r in range(1, max_rows + 1)}):
                cases.append(dict(fixed=fixed, probabilities=probabilities, costs=[dict(rows=r, ms=ms) for r, ms in costs.items()], overhead=0.0, max_rows=max_rows, expected=allocate(fixed, probabilities, costs, 0.0, max_rows)))
    for rates in ([], [0.0], [1.0], [0.94], [0.85, 0.75, 0.7, 0.65], [rng.random() for _ in range(19)]):
        for count in ((0,) if not rates else (0, 1, 3, 15, 32, 128)):
            chains.append(dict(rates=rates, expected=chain_probabilities(rates, count)))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(cases=cases, chains=chains)))
    print(f'Saved {len(cases)} upstream shared allocations and {len(chains)} acceptance chains')


def calibration_fixtures(output):
    import math
    import random
    from tensorfold.drafters import calibration
    from tensorfold.families.qwen3_5.dflash_head import by_chance
    rng = random.Random(161803)
    fits = []
    tables = []
    for depths, scores in [(calibration.DEPTH_EDGES, calibration.SCORE_EDGES), ((-1, 2, 7), (-3.0, -1.0, 0.0)), ((0,), ())]:
        for count in (0, 1, 7, 83, 1000):
            samples = [dict(depth=rng.randrange(-3, 24), score=rng.uniform(-10, 0), landed=rng.choice((True, False))) for _ in range(count)]
            # Opposing adjacent bins force weighted pooling across empty bins as well.
            if count > 1:
                samples += [dict(depth=0, score=-9.0, landed=True)] * 19 + [dict(depth=0, score=-0.001, landed=False)] * 13
            table = calibration.fit(((s['depth'], s['score'], s['landed']) for s in samples), depths, scores)
            raw = dict(depth_edges=table.depth_edges, score_edges=table.score_edges, table=table.table)
            queries = [dict(depth=d, score=s, expected=table.probability(d, s))
                       for d in sorted({-10, 0, 99, *(int(e) + delta for e in depths for delta in (-1, 0, 1))})
                       for s in sorted({-100.0, 0.0, 1.0, *(v for edge in scores for v in (math.nextafter(edge, -math.inf), edge, math.nextafter(edge, math.inf)))})]
            fits.append(dict(samples=samples, expected=raw, serialized=table.as_dict(), queries=queries))
            tables.append(raw)
    shipped = json.loads(Path('src/tensorfold/families/qwen3_5/dflash2_calibration.json').read_text())
    for count in range(65):
        for hits in sorted({0, count // 3, count}):
            samples = [dict(depth=0, score=-1.0, landed=i < hits) for i in range(count)]
            table = calibration.fit(((s['depth'], s['score'], s['landed']) for s in samples), (0,), ())
            raw = dict(depth_edges=table.depth_edges, score_edges=table.score_edges, table=table.table)
            fits.append(dict(samples=samples, expected=raw, serialized=table.as_dict(), queries=[]))
    trees = []
    choices = [None, *tables, *shipped['tables'].values()]
    for size in (0, 1, 2, 7, 15, 31, 63):
        for table in choices:
            for case in range(3):
                parents = [rng.randrange(-1, i) if case == 0 else i - 1 if case == 1 else -1 for i in range(size)]
                tokens = [rng.randrange(248320) for _ in parents]
                scores = [rng.choice((-0.05, -0.5, -1.0, -3.5, -8.0)) for _ in parents]
                chances = calibration.Calibration(**table).probabilities(parents, scores) if table else [math.exp(s) for s in scores]
                ids, qs, ps = by_chance(tokens, parents, chances)
                trees.append(dict(tokens=tokens, parents=parents, scores=scores, table=table, expected=dict(tokens=ids, parents=qs, probabilities=ps)))
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    rounding = {0.0, 1.0, math.nextafter(0, 1), math.nextafter(1, 0)}
    for i in range(10000):
        midpoint = (i + 0.5) / 10000
        rounding.update((math.nextafter(midpoint, -math.inf), midpoint, math.nextafter(midpoint, math.inf)))
    path.write_text(json.dumps(dict(fits=fits, trees=trees, shipped=shipped, rounding=[dict(value=v, expected=round(v, 4)) for v in sorted(rounding)])))
    samples = fits[3]
    path.with_suffix('.samples.json').write_text(json.dumps(dict(source=dict(oracle='upstream'), depth_edges=samples['expected']['depth_edges'], score_edges=samples['expected']['score_edges'], samples=dict(sampled=samples['samples'], greedy=fits[4]['samples']))))
    print(f"Saved {len(fits)} upstream calibration fits and {len(trees)} ranked trees")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="build/models/Qwen3.8-27B-MLX-4bit")
    parser.add_argument("--tokens", help="Exact prompt IDs, also honored during generation")
    parser.add_argument("--dump-logits", type=Path, help="Save the final prefill block before generation")
    parser.add_argument("--metal-sampling", action="store_true")
    parser.add_argument("--simd", action="store_true", help="Use the production SIMD decoder and its MLX attention arithmetic")
    parser.add_argument("--production-kernels", action="store_true", help="Diagnostic: use production loading/fusion with lane-tree prefill (not the serving engine's prefill)")
    parser.add_argument("--disable-fusion", action="store_true", help="Diagnostic: turn off stacked consumers after the production load")
    parser.add_argument("--output", default="build/native-checks/reference.npy")
    parser.add_argument("--compare", nargs=2)
    parser.add_argument("--vision-fixture", nargs=2, type=int, metavar=("GRID_HEIGHT", "GRID_WIDTH"))
    parser.add_argument("--image-fixture", action="store_true", help="Vision fixture dimensions describe a generated RGB PNG before preprocessing")
    parser.add_argument("--image-format", choices=("PNG", "JPEG", "WEBP"), default="PNG")
    parser.add_argument("--image-alpha", action="store_true")
    parser.add_argument("--image-only", action="store_true", help="Generate preprocessing oracle without loading the vision tower")
    parser.add_argument("--image-http-fixtures", action="store_true")
    parser.add_argument("--calibration-fixtures", action="store_true")
    parser.add_argument("--allocation-fixtures", action="store_true")
    parser.add_argument("--memory-fixtures", action="store_true")
    parser.add_argument("--prompt-cache-fixtures", action="store_true")
    parser.add_argument("--prefill-plan-fixtures", action="store_true")
    parser.add_argument("--server-live-fixtures", action="store_true")
    parser.add_argument("--capture-fixtures", action="store_true")
    parser.add_argument("--verify-capture", nargs=2, metavar=('DRAFTER', 'REPORT'))
    parser.add_argument("--compare-calibration", nargs=2, type=Path)
    parser.add_argument("--image-mode", choices=("RGB", "RGBA", "L", "CMYK"))
    parser.add_argument("--image-orientation", type=int, choices=range(1, 9), default=1)
    parser.add_argument("--compare-vision", type=Path)
    parser.add_argument("--compare-arrays", nargs=2, type=Path)
    parser.add_argument("--compare-reports", nargs="+")
    parser.add_argument("--require-rounds", action="store_true", help="Reject trivial EOS-before-decode parity runs")
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--prompt", default="Write a short Python function that computes the Fibonacci sequence.")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    args = parser.parse_args()
    if args.server_live_fixtures:
        return server_live_fixtures(args.output)
    if args.prefill_plan_fixtures:
        return prefill_plan_fixtures(args.output)
    if args.prompt_cache_fixtures:
        return prompt_cache_fixtures(args.output)
    if args.memory_fixtures:
        return memory_fixtures(args.output)
    if args.allocation_fixtures:
        return allocation_fixtures(args.output)
    if args.capture_fixtures:
        return capture_fixtures(args.output)
    if args.verify_capture:
        return verify_capture(args.model, *args.verify_capture)
    if args.calibration_fixtures:
        return calibration_fixtures(args.output)
    if args.compare_calibration:
        from tensorfold.drafters import calibration
        sample_path, output_path = args.compare_calibration
        data = json.loads(sample_path.read_text())
        actual = json.loads(output_path.read_text())
        expected = {name: calibration.fit(((s['depth'], s['score'], s['landed']) for s in samples), data['depth_edges'], data['score_edges']).as_dict() for name, samples in data['samples'].items()}
        assert actual == dict(source=data['source'], tables=expected), 'Native fitted calibration file differs from upstream'
        print('PASS: native calibration CLI output reloads with exact upstream table values and source metadata')
        return
    if args.image_http_fixtures:
        return image_http_fixtures(args.output)
    if args.vision_fixture:
        return vision_fixture(args.model, args.output, *args.vision_fixture, args.image_fixture, args.image_format, args.image_alpha, args.image_orientation, args.image_only, args.image_mode)
    if args.compare_vision or args.compare_arrays:
        reference, actual = args.compare_arrays or (args.compare_vision / "python", args.compare_vision / "native")
        files = sorted(reference.glob("*.npy"))
        assert files, "No oracle arrays"
        failed = []
        for path in files:
            a = np.load(path)
            b = np.load(actual / path.name)
            equal = a.shape == b.shape and np.array_equal(a, b)
            if not equal or len(files) <= 64:
                difference = np.max(np.abs(a - b)) if a.shape == b.shape else "shape mismatch"
                print(f"{path.stem}: {'PASS' if equal else 'FAIL'}, max difference {difference}")
            if not equal:
                failed.append(path.stem)
        assert not failed, failed
        print(f"PASS: {len(files)} arrays bit-exact")
        return
    if args.compare_reports:
        reports = [json.loads(Path(p).read_text()) for p in args.compare_reports]
        if args.require_rounds:
            for report in reports:
                assert report["rounds"] > 0 and len(report["tokens"]) > 1, "No decode rounds exercised"
        for report in reports[1:]:
            assert report["prompt_tokens"] == reports[0]["prompt_tokens"], "Tokenizer mismatch"
            assert report["tokens"] == reports[0]["tokens"], "Output token mismatch"
        print(f"PASS: {len(reports)} reports have identical prompts and {len(reports[0]['tokens'])} output token IDs")
        return
    if args.compare:
        a, b = [np.load(p) for p in args.compare]
        if b.dtype.kind == "V" and b.dtype.itemsize == 2:
            b = (b.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
        print(f"shapes: {a.shape} / {b.shape}")
        print(f"equal elements: {np.count_nonzero(a == b)}/{a.size}")
        print(f"max absolute difference: {np.max(np.abs(a - b))}")
        if a.size // a.shape[-1] <= 32:
            print(f"argmax reference: {a.argmax(axis=-1).tolist()}")
            print(f"argmax native: {b.argmax(axis=-1).tolist()}")
        if not np.array_equal(a, b):
            raise SystemExit(1)
        return
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from tensorfold.families.qwen3_5 import load_lane_model
    from tensorfold.kernels.qwen.dense.v1 import lane_qmm, lane_tree

    bonsai = json.loads((Path(args.model) / "config.json").read_text()).get("model_type") == "prism_hadamard_qwen35"
    if bonsai:
        from tensorfold.families.bonsai import pack
        from mlx_lm.utils import load_tokenizer
        model = pack.build(Path(args.model), form="packed" if args.simd else "lanes")
        tokenizer = load_tokenizer(Path(args.model))
        if args.simd:
            from tensorfold.kernels.qwen.dense.v1 import row_forward
            from tensorfold.families.qwen3_5 import install_row_decoder
            if not install_row_decoder(model):
                raise ValueError("Bonsai row decoder rejected this checkpoint")
            tree_forward, commit_tree = row_forward.forward, row_forward.commit
    elif args.simd:
        if args.production_kernels or args.disable_fusion:
            parser.error("--simd uses the production decoder without fusion overrides")
        from tensorfold.families.qwen3_5 import load
        from tensorfold.kernels.qwen.dense.v1 import row_forward
        family, tokenizer = load(Path(args.model), lane_kernels="off")
        model = family.inner
        tree_forward, commit_tree = row_forward.forward, row_forward.commit
    elif args.production_kernels:
        from tensorfold.families.qwen3_5 import load
        family, tokenizer = load(Path(args.model))
        model = family.inner
        if args.disable_fusion:
            from tensorfold.kernels.qwen.dense.v1 import lane_fuse
            lane_fuse.enabled = False
    else:
        if args.disable_fusion:
            parser.error("--disable-fusion requires --production-kernels")
        model, tokenizer = load_lane_model(Path(args.model))
    if not args.simd:
        tree_forward, commit_tree = lane_tree.tree_forward, lane_tree.commit_tree
        if not args.production_kernels:
            lane_qmm.install(model, rows=128, tile=True, wide=True)
    core = model.language_model.model
    head = model.language_model.lm_head
    tokens = ([int(x) for x in args.tokens.split(",")] if args.tokens else
              tokenizer.encode(args.prompt, add_special_tokens=False) if args.generate else [1, 2, 3, 4])
    if not tokens:
        raise ValueError("Empty prompt")
    cache = make_prompt_cache(model)
    # Same 128-row grid as the native CLI; commit the whole prompt before decoding.
    for start in range(0, len(tokens), 128):
        block = tokens[start:start + 128]
        if args.simd:
            # Production SIMD verification uses windows of at most 16 rows;
            # wider calls switch to regular prompt attention arithmetic.
            parts = []
            for offset in range(0, len(block), 16):
                rows = block[offset:offset+16]
                part, record = tree_forward(core, head, rows, list(range(-1, len(rows)-1)), cache, start + offset)
                mx.eval(part)
                commit_tree(cache, record, list(range(len(rows))), len(rows), start + offset)
                parts.append(part)
            logits = mx.concatenate(parts, axis=1)
        else:
            logits, record = tree_forward(core, head, block, list(range(-1, len(block)-1)), cache, start)
            mx.eval(logits)
            commit_tree(cache, record, list(range(len(block))), len(block), start)
        if start % 1024 == 0:
            print(f"Prefill {start + len(block)}/{len(tokens)}", flush=True)
    if args.dump_logits:
        args.dump_logits.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.dump_logits, np.array(logits.astype(mx.float32)))
    if args.generate:
        from tensorfold.engine.exact_sampling import Sampling, sample_rows
        settings = Sampling(args.seed, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p)
        def select(logits, position):
            last = logits[0, -1:]
            if args.metal_sampling:
                from tensorfold.engine.gpu_sampling import sample
                return int(sample(last, settings if args.temperature else None, [position]).item())
            return sample_rows(last, [position], settings)[0] if args.temperature else int(mx.argmax(last).item())
        position = len(tokens)
        pending = select(logits, position)
        generated = [pending]
        while len(generated) < args.generate and pending not in (248044, 248046):
            logits, record = tree_forward(core, head, [pending], [-1], cache, position)
            commit_tree(cache, record, [0], 1, position)
            position += 1
            pending = select(logits, position)
            generated.append(pending)
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        sha = hashlib.sha256(np.array(generated, dtype="<u4").tobytes()).hexdigest()
        out.write_text(json.dumps(dict(prompt_tokens=tokens, tokens=generated, token_sha256=sha,
                                      peak_mlx_bytes=mx.get_peak_memory(), active_mlx_bytes=mx.get_active_memory())))
        print(f"Saved {out}: {len(generated)} tokens, SHA-256 {sha}")
        return
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.array(logits.astype(mx.float32)))
    print(f"Saved {out}: shape={logits.shape}, argmax={mx.argmax(logits, axis=-1).tolist()}")


if __name__ == "__main__":
    main()
