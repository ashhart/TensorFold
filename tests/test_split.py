"""The split's checkpoint units, weight cache and stage sessions, over loopback TCP with a fake stage backend."""

from __future__ import annotations

import json
import socket
import struct
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from tensorfold.split import checkpoint, client, stage, wire
from tensorfold.split.store import Store

LAYERS, H = 4, 8


def write_safetensors(path: Path, tensors: dict[str, np.ndarray]) -> None:
    header, at, blobs = {}, 0, []
    for name, a in tensors.items():
        raw = a.tobytes()
        header[name] = {"dtype": {np.float32: "F32", np.int32: "I32"}[a.dtype.type], "shape": list(a.shape),
                        "data_offsets": [at, at + len(raw)]}
        at += len(raw)
        blobs.append(raw)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


@pytest.fixture
def model(tmp_path, monkeypatch) -> Path:
    monkeypatch.setattr(checkpoint, "SIDECAR", tmp_path / "digests.json")
    d = tmp_path / "model"
    d.mkdir()
    rng = np.random.default_rng(0)
    p = "language_model.model."
    first = {p + "embed_tokens.weight": rng.standard_normal((16, H), dtype=np.float32)}
    second = {}
    for i in range(LAYERS):
        target = first if i < 2 else second
        target[f"{p}layers.{i}.mlp.up_proj.weight"] = rng.standard_normal((H, H), dtype=np.float32)
        target[f"{p}layers.{i}.input_layernorm.weight"] = rng.standard_normal((H,), dtype=np.float32)
    second[p + "norm.weight"] = np.ones((H,), dtype=np.float32)
    second["language_model.lm_head.weight"] = rng.standard_normal((16, H), dtype=np.float32)
    second["mtp.layers.0.mlp.up_proj.weight"] = np.zeros((H, H), dtype=np.float32)
    write_safetensors(d / "model-00001-of-00002.safetensors", first)
    write_safetensors(d / "model-00002-of-00002.safetensors", second)
    (d / "config.json").write_text(json.dumps({"model_type": "fake", "text_config": {"num_hidden_layers": LAYERS}}))
    return d


def test_units_hold_the_stage_layers_and_the_final_norm(model):
    units = checkpoint.stage_units(model, 2, LAYERS, LAYERS)
    assert [u.label for u in units] == ["layer.2", "layer.3", "final"]
    assert {t.name for u in units for t in u.tensors} == {
        "language_model.model.layers.2.mlp.up_proj.weight", "language_model.model.layers.2.input_layernorm.weight",
        "language_model.model.layers.3.mlp.up_proj.weight", "language_model.model.layers.3.input_layernorm.weight",
        "language_model.model.norm.weight"}                            # no head, no embedding, no MTP head
    assert checkpoint.layer_bytes(model, 2) == sum(u.nbytes for u in units[:2])
    assert [u.key for u in units] == [u.key for u in checkpoint.stage_units(model, 2, LAYERS, LAYERS)]


def test_a_changed_shard_changes_its_units_keys(model, tmp_path):
    before = {u.label: u.key for u in checkpoint.stage_units(model, 1, LAYERS, LAYERS)}
    shard = model / "model-00002-of-00002.safetensors"
    raw = bytearray(shard.read_bytes())
    raw[-1] ^= 1
    time.sleep(0.01)
    shard.write_bytes(bytes(raw))
    after = {u.label: u.key for u in checkpoint.stage_units(model, 1, LAYERS, LAYERS)}
    assert before["layer.1"] == after["layer.1"]                      # its shard did not change
    assert before["layer.2"] != after["layer.2"] and before["final"] != after["final"]


def test_stage_ranges_are_checked(model):
    for first, last in ((0, LAYERS), (2, 2), (1, LAYERS + 1)):
        with pytest.raises(ValueError):
            checkpoint.stage_units(model, first, last, LAYERS)


def test_store_writes_units_a_loader_reads_back(model, tmp_path):
    store = Store(tmp_path / "cache")
    units = checkpoint.stage_units(model, 2, LAYERS, LAYERS)
    described = [u.describe() for u in units]
    assert store.missing(described) == [u.key for u in units]
    for u in units:
        buf = bytearray()
        checkpoint.read_unit(u, buf.extend)
        view = memoryview(bytes(buf))
        at = 0

        def read(n, view=view):
            nonlocal at
            chunk = view[at:at + n]
            at += len(chunk)
            return bytes(chunk)

        store.receive(u.describe(), read)
    assert store.missing(described) == []
    where = store.stage({"model_type": "fake"}, described)
    files = sorted(where.glob("*.safetensors"))
    assert len(files) == 3 and all(f.is_symlink() for f in files)
    for u in units:
        base, header = checkpoint.read_header(store.path(u.key))
        assert base % 4096 == 0                                      # the data starts on a page
        for t in u.tensors:
            begin, end = header[t.name]["data_offsets"]
            with open(t.path, "rb") as f:
                f.seek(t.offset)
                assert store.path(u.key).read_bytes()[base + begin:base + end] == f.read(t.nbytes)
    index = json.loads((where / "model.safetensors.index.json").read_text())["weight_map"]
    assert set(index) == {t.name for u in units for t in u.tensors}
    assert store.stage({"model_type": "fake"}, described) == where


class FakeStage:
    """Adds the start position to each row it receives and returns the rows as its normed output."""

    loads = 0

    def __init__(self, model_dir, first, last, layers, options):
        FakeStage.loads += 1
        self.hidden, self.tap_layers = H, ()
        self.dir = Path(model_dir)

    def reset(self):
        pass

    def close(self):
        pass

    def handle(self, head, body):
        if head["op"] == wire.FETCH:             # a reply in parts, as a stage hands state over
            return [wire.REPLY.pack(0, head["rows"], 0), np.full(head["rows"], head["extra"], dtype=np.uint16),
                    np.arange(head["layer"], dtype=np.uint16)]
        rows = head["rows"]
        at = 4 * head["n_commit"] + (4 * head["extra"] if head["op"] == wire.FORWARD else 0)
        x = np.frombuffer(body, dtype=np.uint16, count=rows * H, offset=at)
        return wire.REPLY.pack(0, rows, 0) + (x + head["start"]).astype(np.uint16).tobytes()


@pytest.fixture
def server(tmp_path, monkeypatch):
    monkeypatch.setattr(stage, "backend_class", lambda kind: FakeStage)
    FakeStage.loads = 0
    srv = stage.Server("127.0.0.1:0", str(tmp_path / "stage-cache"), log=lambda s: None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    for _ in range(200):
        if srv.listener is not None:
            break
        time.sleep(0.01)
    srv.address_str = "127.0.0.1:%d" % srv.listener.getsockname()[1]
    return srv


def rows_call(session, start, rows):
    x = (np.arange(rows * H, dtype=np.uint16) % 1000).reshape(rows, H)
    head = wire.request(wire.PREFILL, epoch=session.epoch, rows=rows, start=start, extra=0, n_commit=0)
    session.channel.submit([head, x, np.zeros_like(x)])
    reply = session.channel.result(timeout=10)
    status, n, _ = wire.REPLY.unpack_from(reply, 0)
    return status, np.frombuffer(bytes(reply[wire.REPLY.size:]), dtype=np.uint16).reshape(n, H) - x if not status else bytes(reply[12:])


def test_a_session_pushes_once_and_serves_rows(model, server):
    s = client.open_session(server.address_str, model, 2, log=lambda m: None)
    assert s.pushed == checkpoint.layer_bytes(model, 2) + 4 * H and s.hidden == H
    status, delta = rows_call(s, 5, 3)
    assert status == 0 and (delta == 5).all()
    s.sock.close()
    again = client.open_session(server.address_str, model, 2, log=lambda m: None)   # cached: nothing moves
    assert again.pushed == 0 and FakeStage.loads == 1                 # and the loaded layers are kept
    assert rows_call(again, 7, 2)[0] == 0
    again.sock.close()


def test_a_new_session_replaces_a_live_one(model, server):
    old = client.open_session(server.address_str, model, 2, log=lambda m: None)
    new = client.open_session(server.address_str, model, 2, log=lambda m: None)   # the old Mac never said goodbye
    assert rows_call(new, 1, 1)[0] == 0
    with pytest.raises((ConnectionError, OSError, socket.timeout)):
        rows_call(old, 1, 1)


def test_a_request_of_another_epoch_is_refused(model, server):
    s = client.open_session(server.address_str, model, 2, log=lambda m: None)
    s.epoch += 2
    status, text = rows_call(s, 0, 1)
    assert status == 1 and b"epoch" in text


def test_an_unknown_family_is_refused_before_weights_move(model, tmp_path):
    srv = stage.Server("127.0.0.1:0", str(tmp_path / "c2"), log=lambda s: None)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    while srv.listener is None:
        time.sleep(0.01)
    with pytest.raises((RuntimeError, ConnectionError)):
        client.open_session("127.0.0.1:%d" % srv.listener.getsockname()[1], model, 2, log=lambda m: None)
    assert not list((tmp_path / "c2" / "units").iterdir())


def test_requests_carry_the_layer_their_rows_enter_at():
    head = wire.parse_request(wire.request(wire.PREFILL, epoch=5, rows=3, start=7, extra=1, n_commit=0, layer=16))
    assert (head["op"], head["layer"], head["epoch"], head["rows"], head["start"]) == (wire.PREFILL, 16, 5, 3, 7)
    assert wire.parse_request(wire.request(wire.FORWARD, epoch=1, rows=1, start=0, extra=1, n_commit=0))["layer"] == 0


def test_a_stage_reply_in_parts_arrives_whole(model, server):
    s = client.open_session(server.address_str, model, 2, log=lambda m: None)
    s.channel.submit([wire.request(wire.FETCH, epoch=s.epoch, rows=4, start=0, extra=9, n_commit=0, layer=3)])
    reply = s.channel.result(timeout=10)
    status, n, _ = wire.REPLY.unpack_from(reply, 0)
    body = np.frombuffer(bytes(reply[wire.REPLY.size:]), dtype=np.uint16)
    assert status == 0 and n == 4 and body.tolist() == [9, 9, 9, 9, 0, 1, 2]


def fake_sysfs(tmp_path):
    from tensorfold.split import rdma

    port = tmp_path / "ib" / "rocep1s0f1" / "ports" / "1"
    gids = {0: ("fe80:0000:0000:0000:4ebb:47ff:fe7d:a1a5", "IB/RoCE v1"),
            1: ("fe80:0000:0000:0000:4ebb:47ff:fe7d:a1a5", "RoCE v2"),
            2: ("0000:0000:0000:0000:0000:ffff:c0a8:c802", "IB/RoCE v1"),
            4: ("0000:0000:0000:0000:0000:ffff:c0a8:c802", "RoCE v2")}       # a flap left index 3 empty
    for d in ("gids", "gid_attrs/types", "gid_attrs/ndevs"):
        (port / d).mkdir(parents=True)
    for i in range(6):
        gid, kind = gids.get(i, ("0000:0000:0000:0000:0000:0000:0000:0000", ""))
        (port / "gids" / str(i)).write_text(gid + "\n")
        (port / "gid_attrs/types" / str(i)).write_text(kind + "\n")
        (port / "gid_attrs/ndevs" / str(i)).write_text("mac-rdma-bond\n")
    (tmp_path / "net" / "mac-rdma-bond").mkdir(parents=True)
    (tmp_path / "net" / "mac-rdma-bond" / "address").write_text("4c:bb:47:7d:a1:a5\n")
    return rdma, tmp_path / "ib", tmp_path / "net"


def test_the_stage_finds_its_roce_gid_by_address_not_index(tmp_path):
    rdma, ib, net = fake_sysfs(tmp_path)
    a = rdma.roce_address("rocep1s0f1", sysfs=ib, net=net)
    assert (a.gid_index, a.ip, a.mac, a.netdev) == (4, "192.168.200.2", "4c:bb:47:7d:a1:a5", "mac-rdma-bond")
    with pytest.raises(ValueError, match="no RoCE v2 IPv4 GID for 10.0.0.1"):
        rdma.roce_address("rocep1s0f1", "10.0.0.1", sysfs=ib, net=net)
    with pytest.raises(ValueError, match="no RDMA device"):
        rdma.roce_address("mlx5_9", sysfs=ib, net=net)


def test_the_mac_takes_the_other_end_of_a_point_to_point_subnet():
    from tensorfold.split.rdma import peer_ip

    assert peer_ip("192.168.200.2", 30) == "192.168.200.1"
    assert peer_ip("10.0.0.0", 31) == "10.0.0.1"
    with pytest.raises(ValueError, match="point-to-point"):
        peer_ip("192.168.200.2", 24)
