"""The HF reference of the Ling oracles: inclusionAI/Ling-3.1-flash at one revision, its files by sha256."""
import hashlib
import os

REPO = "inclusionAI/Ling-3.1-flash"
# config.json is this revision's (#2); the modeling, configuration and index files are unchanged since the upload
REVISION = "a6943f4bf04f8041ad983d2b58dd43b5ba86ae4a"
SHA256 = {
    "modeling_bailing_moe_v3.py": "c2509bf7ac580c262e2581d34d6403aa21682d2e10beb9ad85ad8820a7e33a40",
    "configuration_bailing_moe_v3.py": "f2c048966aec8a2f042cfeb1351f74d51a28589b409c55baae7d24e841c1f6c4",
    "config.json": "6f3773e52a12509e772e698aacacaa156d2402ebb8250812a1da52a03d92f084",
    "model.safetensors.index.json": "f01006f7a65eb4b32b66ba60868ef7cc1642c66afe658cb7e6faf80eb246b468",
}
MODELING = ("modeling_bailing_moe_v3.py", "configuration_bailing_moe_v3.py")


def pinned(folder: str, files=MODELING) -> None:
    """Stops unless each of `files` in `folder` is the pinned revision's, byte for byte."""
    for f in files:
        path = os.path.join(folder, f)
        with open(path, "rb") as fh:
            got = hashlib.sha256(fh.read()).hexdigest()
        if got != SHA256[f]:
            raise SystemExit(f"{path}: sha256 {got}, not {REPO}@{REVISION[:12]}'s {SHA256[f]}")
