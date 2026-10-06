"""The public EXL3 source-view recipe must preserve the pinned BF16 source."""
import json
import subprocess
import sys
from pathlib import Path


def test_nemotron_exl3_source_view_is_non_mutating_and_repeatable(tmp_path):
    source = tmp_path / "bf16"
    overlay = tmp_path / "view"
    source.mkdir()
    config = {
        "architectures": ["NemotronHForCausalLM"], "model_type": "nemotron_h", "dtype": "bfloat16",
        "num_hidden_layers": 3, "layers_block_type": ["mamba", "moe", "attention"],
        "num_nextn_predict_layers": 1, "mtp_layers_block_type": ["attention", "moe"],
    }
    original = json.dumps(config).encode()
    (source / "config.json").write_bytes(original)
    (source / "tokenizer.json").write_text("{}")
    suffixes = ["backbone.layers.0.mixer.in_proj.weight",
                "backbone.layers.0.mixer.conv1d.weight",
                "backbone.layers.1.mixer.experts.0.up_proj.weight",
                "backbone.layers.1.mixer.experts.0.down_proj.weight",
                "backbone.layers.2.mixer.q_proj.weight",
                "backbone.layers.2.mixer.k_proj.weight",
                "mtp.layers.0.eh_proj.weight", "mtp.layers.0.mixer.q_proj.weight",
                "mtp.layers.1.mixer.experts.0.up_proj.weight",
                "mtp.layers.1.final_layernorm.weight"]
    shard_names = [f"model-{i:05d}-of-00014.safetensors" for i in range(1, 15)]
    index = {key: shard_names[i % len(shard_names)] for i, key in enumerate(suffixes)}
    index.update({f"other.{i}": shard_names[i] for i in range(len(shard_names))})
    (source / "model.safetensors.index.json").write_text(json.dumps({"weight_map": index}))
    script = Path(__file__).resolve().parents[1] / "tools/prepare_nemotron_exl3_source.py"
    command = [sys.executable, str(script), "--source", str(source), "--overlay", str(overlay)]
    for _ in range(2):
        run = subprocess.run(command, text=True, capture_output=True)
        assert run.returncode == 0, run.stderr
    assert (source / "config.json").read_bytes() == original
    view = json.loads((overlay / "config.json").read_text())
    assert view["hybrid_override_pattern"] == "ME*"
    assert view["mtp_hybrid_override_pattern"] == "*E"
    assert view["layers_block_type"] == ["linear_attention", "moe", "full_attention"]
    assert (overlay / "tokenizer.json").is_symlink()
    assert (overlay / "tokenizer.json").resolve() == (source / "tokenizer.json").resolve()
