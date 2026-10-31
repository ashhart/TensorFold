from tensorfold_web import machine


def test_a_gb10_is_unified_memory_though_nvidia_smi_gives_none(monkeypatch):
    # the GX10 of 10 Oct 2026: nvidia-smi says "[N/A]" for the GB10's memory, the firmware names the machine
    monkeypatch.setattr(machine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(machine.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(machine, "_query", lambda cmd: ("NVIDIA GB10, [N/A], 12.1\n", None))
    monkeypatch.setattr(machine.shutil, "which", lambda name: "/usr/bin/" + name)   # as if nvidia-smi were there
    monkeypatch.setattr(machine, "_product_name", lambda: "GX10")
    monkeypatch.setattr(machine, "_tensorfold_version", lambda: None)
    machine._reset()
    try:
        m = machine._detect()
    finally:
        machine._reset()
    assert m.backend == "cuda" and m.unified_memory is True and m.chip == "NVIDIA GB10 (GX10)"
    assert m.gpus[0].memory_bytes is None


def test_cuda_floor_follows_the_native_archive():
    from tensorfold_web import machine

    assert machine.cuda_floor("x86_64") == (8, 6)          # RTX 3090 and the other Ampere cards
    assert machine.cuda_floor("aarch64") == (8, 9)


def test_a_stalled_nvidia_smi_is_asked_again_rather_than_remembered_as_no_gpu(monkeypatch):
    monkeypatch.setattr(machine.platform, "system", lambda: "Linux")
    monkeypatch.setattr(machine.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(machine.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(machine, "_product_name", lambda: "GX10")
    monkeypatch.setattr(machine, "_tensorfold_version", lambda: None)
    answers = [("", "nvidia-smi did not answer within 5 s"), ("", "nvidia-smi did not answer within 5 s"),
               ("NVIDIA GB10, [N/A], 12.1\n", None)]
    asked = []
    monkeypatch.setattr(machine, "_query", lambda cmd: asked.append(cmd[0]) or answers[len(asked) - 1])
    machine._reset()
    try:
        first = machine._detect()
        assert first.backend == "none" and "nvidia-smi did not answer" in first.backend_note
        assert machine._detect().backend == "none" and len(asked) == 2      # asked again, not remembered
        assert machine._detect().backend == "cuda" and len(asked) == 3
        assert machine._detect().backend == "cuda" and len(asked) == 3      # a good answer is kept
    finally:
        machine._reset()
