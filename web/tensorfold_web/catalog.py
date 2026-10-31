"""The catalog: curated entries, their size on Hugging Face, whether they are downloaded, and their memory verdict."""

from __future__ import annotations

import json
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from . import fit
from .machine import Machine

DATA = Path(__file__).parent / "data" / "catalog.json"
CACHE_TTL = 24 * 3600

FACTS_RETRY_S = 300            # a checkpoint the engine's `info` refused is asked again after this long


@dataclass
class Entry:
    id: str
    family: str
    name: str
    vendor: str
    license: str
    description: str
    tags: list[str]
    backends: list[str]
    drafter: str | None
    drafter_optional: str | None
    max_context: int
    bytes_per_token: int
    size_bytes: int | None
    speed_hint: str
    fraction: float | None = None
    workspace_bytes: int | None = None
    drafter_fraction: float = 1.0        # share of the drafter's files that stays resident (bf16 files loaded at 4 bits: 0.3)
    drafter_bytes: int | None = None
    drafter_size_bytes: int | None = None  # the drafter's files as last measured: its weight before sizes are fetched
    native: bool = False                 # served by the native engine: what its `capabilities` say on this machine,
                                         # this flag only when no engine is installed to ask
    format: str | None = None            # the checkpoint's weight format as the engine names it ("mlx-q4g64")
    native_from: str | None = None       # a mixed-width (oQ) checkpoint: the first native release that serves it, as
                                         # `capabilities --json` lists each family's uniform formats only
    native_overhead_bytes: int | None = None   # what the native engine holds above `info`'s floor at idle, if measured
    engine_info: dict | None = None      # what the native engine's `info` says of the downloaded checkpoint
    installed: bool = False
    verdict: dict | None = None
    runs_here: bool = True
    runs_note: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _load() -> list[Entry]:
    data = json.loads(DATA.read_text())
    return [Entry(**{k: v for k, v in m.items() if k in Entry.__dataclass_fields__}) for m in data["models"]]


def _sizes_cache_path() -> Path:
    from .settings import config_dir

    return config_dir() / "sizes.json"


def _read_sizes_cache() -> dict[str, Any]:
    try:
        return json.loads(_sizes_cache_path().read_text())
    except (OSError, ValueError):
        return {}


def _write_sizes_cache(cache: dict[str, Any]) -> None:
    path = _sizes_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache, indent=1))
    except OSError:
        pass


def repo_size(repo_id: str, *, timeout: float = 15.0) -> int | None:
    """Bytes of every file in a repository on Hugging Face (what a snapshot download fetches), or None when offline."""

    try:
        from huggingface_hub import HfApi

        info = HfApi().model_info(repo_id, files_metadata=True, timeout=timeout)
    except Exception:  # noqa: BLE001 - offline, gated, moved: the curated size stands
        return None
    total = sum(int(s.size) for s in info.siblings or [] if s.size)
    return total or None


def installed(repo_id: str, expected_bytes: int | None = None) -> bool:
    """Whether the checkpoint is in the Hugging Face cache, whole: the cache answers as soon as `refs/main` exists,
    so the snapshot's files are weighed against what the catalog says (a download cut halfway is lighter); with no
    known weight, an `.incomplete` blob is the sign of a cut download."""

    try:
        from huggingface_hub import snapshot_download

        from . import pull

        snapshot = Path(snapshot_download(repo_id, local_files_only=True))
        if expected_bytes:
            have = 0
            for f in snapshot.rglob("*"):
                if f.is_symlink() and not f.exists():          # a link to a blob that never finished
                    return False
                if f.is_file():
                    have += f.stat().st_size                   # through the link: the blob's real size
            return have >= 0.98 * expected_bytes
        blobs = pull.blobs_dir(repo_id)
        return not (blobs.is_dir() and any(blobs.glob("*.incomplete")))
    except Exception:  # noqa: BLE001
        return False


def capacity_of(machine: Machine) -> fit.Capacity | None:
    """What this machine can give an engine, by backend; None when no supported GPU is present."""

    if machine.backend == "mlx":
        return fit.mlx_capacity(machine.ram_bytes, machine.gpu_ceiling_bytes)
    if machine.backend == "cuda":
        if machine.unified_memory:                 # a GB10 and its relatives: the GPU draws on the RAM
            return fit.cuda_capacity(machine.ram_bytes, unified=True)
        if machine.gpus and machine.gpus[0].memory_bytes:
            return fit.cuda_capacity(int(machine.gpus[0].memory_bytes), unified=False)
        return None                                # a card whose memory is not known: no verdict rather than the RAM's
    return None


class Catalog:
    """Entries with sizes refreshed in the background and verdicts computed for a machine and a budget fraction."""

    def __init__(self) -> None:
        self.entries = _load()
        self._lock = threading.Lock()
        self._sizes = _read_sizes_cache()
        self._refreshing = False
        self.refreshed_at: float | None = None
        self._facts: dict[str, dict] = {}            # `info` by checkpoint, asked once it is downloaded
        self._facts_missed: dict[str, float] = {}    # when `info` last gave nothing for one: not asked again for a while

    def refresh_sizes(self, *, force: bool = False) -> None:
        now = time.time()
        with self._lock:
            if self._refreshing:
                return
            self._refreshing = True
        try:
            cache = dict(self._sizes)
            repos = set()
            for e in self.entries:
                repos.add(e.id)
                if e.drafter:
                    repos.add(e.drafter)
            for repo in sorted(repos):
                hit = cache.get(repo)
                if not force and hit and now - hit.get("at", 0) < CACHE_TTL:
                    continue
                size = repo_size(repo)
                if size:
                    cache[repo] = {"bytes": size, "at": now}
            with self._lock:
                self._sizes = cache
                self.refreshed_at = time.time()
            _write_sizes_cache(cache)
        finally:
            with self._lock:
                self._refreshing = False

    @property
    def refreshing(self) -> bool:
        return self._refreshing

    def refresh_sizes_async(self) -> None:
        threading.Thread(target=self.refresh_sizes, name="catalog-sizes", daemon=True).start()

    def size_of(self, repo: str | None) -> int | None:
        if not repo:
            return None
        hit = self._sizes.get(repo)
        return int(hit["bytes"]) if hit else None

    def weights_of(self, e: Entry) -> int:
        """Bytes the estimate counts as resident: the checkpoint plus the drafter's resident share."""

        size = self.size_of(e.id) or e.size_bytes or 0
        # the drafter's measured size when the hub's is not known yet (a first start offline): not counted as nothing
        drafter = int((self.size_of(e.drafter) or e.drafter_size_bytes or 0) * e.drafter_fraction) if e.drafter else 0
        return size + drafter

    def verdict_of(self, e: Entry, machine: Machine, weights: int, workspace: int, fraction: float | None,
                   *, strict: bool = False) -> fit.Verdict:
        """The verdict by the rule of the engine that serves the model: the native engine on a Mac sizes its own
        memory up to macOS's GPU limit, the budget setting does not apply to it; the Python line keeps its rule."""

        if e.native and machine.backend == "mlx":
            cap = fit.native_capacity(machine.ram_bytes, machine.gpu_ceiling_bytes)
            return fit.verdict(cap, weights, bytes_per_token=e.bytes_per_token, max_context=e.max_context,
                               workspace_bytes=workspace + (e.native_overhead_bytes or fit.NATIVE_OVERHEAD),
                               copies=fit.NATIVE_COPIES)
        cap = capacity_of(machine)
        assert cap is not None
        return fit.verdict(cap, weights, bytes_per_token=e.bytes_per_token, max_context=e.max_context,
                           workspace_bytes=workspace, model_fraction=e.fraction, requested_fraction=fraction,
                           strict=strict)

    def fit_inputs(self, e: Entry) -> tuple[int, int, bool]:
        """Resident bytes and working room for the verdict: the engine's own `info` once a native checkpoint is
        downloaded (its weights, and its memory floor less them as the runtime margin), the catalog's estimate
        otherwise. The third value says which."""

        facts = self._facts.get(e.id) if e.native and not e.drafter else None
        if facts and facts.get("weights_bytes") and facts.get("floor_bytes"):
            return facts["weights_bytes"], max(0, facts["floor_bytes"] - facts["weights_bytes"]), True
        return self.weights_of(e), e.workspace_bytes or fit.DEFAULT_WORKSPACE, False

    def engine_facts(self, e: Entry) -> dict | None:
        """The native engine's `info` on a downloaded checkpoint it serves: family, format, context, memory floor."""

        if not e.native:
            return None
        if e.id not in self._facts:
            from . import native

            binary = native.find_binary()
            if binary is None:
                return None
            if time.time() - self._facts_missed.get(e.id, 0) < FACTS_RETRY_S:
                return None                           # refused or hung a moment ago: not run again on every view
            try:
                facts = native.info(binary, e.id)
            except Exception:  # noqa: BLE001 - a binary that hangs or fails says nothing
                facts = None
            if facts is None:                         # asked again later: the checkpoint may complete, the engine change
                self._facts_missed[e.id] = time.time()
                return None
            self._facts[e.id] = facts
        return self._facts[e.id]

    def native_here(self, e: Entry) -> bool:
        """Whether the native engine installed here serves ``e``: its family and format among those the engine's
        `capabilities --json` lists, unless it refused to load it here before. Without an engine to ask, the catalog's own
        flag."""

        from . import native

        binary = native.find_binary()
        if binary is None or not (e.format or e.native_from):
            return e.native
        if native.refused(binary, e.id):            # listed by family, refused on this chip when it was tried
            return False
        try:
            families = native.capabilities(binary).get("families") or {}
        except Exception:  # noqa: BLE001 - an engine that cannot say keeps the catalog's flag
            return e.native
        if e.native_from:                           # its family served, by a release that reads mixed widths
            return e.family in families and native.version_tuple(native.version_of(binary)) >= \
                native.version_tuple(e.native_from)
        return e.format in (families.get(e.family) or [])

    def view(self, machine: Machine, *, fraction: float | None = None, backend: str | None = None) -> list[dict]:
        """Every entry with its size, installed flag and verdict for ``machine`` at the requested budget fraction."""

        backend = backend or machine.backend
        cap = capacity_of(machine)
        python_line: bool | None = None
        out = []
        for e in self.entries:
            e = Entry(**asdict(e))
            e.native = self.native_here(e)
            e.size_bytes = self.size_of(e.id) or e.size_bytes
            e.drafter_bytes = self.size_of(e.drafter) if e.drafter else None
            e.installed = installed(e.id, e.size_bytes) and (not e.drafter or installed(e.drafter, e.drafter_bytes))
            e.runs_here = backend in e.backends
            if e.installed:
                e.engine_info = self.engine_facts(e)
            if not e.runs_here:
                e.runs_note = (f"Runs on {' and '.join(b.upper() for b in e.backends)} machines; this one has "
                               f"{'no supported GPU' if backend == 'none' else backend.upper()}.")
            elif not e.native:
                if python_line is None:
                    from .engines import PythonLauncher

                    python_line = PythonLauncher().available()
                if not python_line:                     # neither engine here serves it
                    e.runs_here = False
                    e.runs_note = ("The native engine installed here does not serve this model yet, and the Python "
                                   "line (0.6.6) that does is not installed.")
            weights, workspace, from_engine = self.fit_inputs(e)
            if weights > 0 and cap is not None:
                v = self.verdict_of(e, machine, weights, workspace, fraction)
                if from_engine and v.basis == "mlx-calibrated":    # CUDA's budget rule stays unverified either way
                    v.basis = "engine-floor"
                e.verdict = v.to_dict()
            out.append(e.to_dict())
        # one model to point a newcomer at: it runs here and fits, the fastest first, one already on disk before
        # one to download
        def rank(m: dict) -> tuple:
            level = (m.get("verdict") or {}).get("level")
            return (m["runs_here"] and level in ("fits", "tight"), level == "fits", bool(m.get("installed")),
                    {"fastest": 2, "fast": 1}.get(m.get("speed_hint"), 0), "chat" in (m.get("tags") or []))
        best = max(out, key=rank, default=None)
        if best is not None and rank(best)[0]:
            best["recommended"] = True
        return out
